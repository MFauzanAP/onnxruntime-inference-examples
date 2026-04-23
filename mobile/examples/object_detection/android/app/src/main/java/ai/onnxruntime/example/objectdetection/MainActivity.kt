package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.example.objectdetection.databinding.ActivityMainBinding
import java.io.ByteArrayOutputStream
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private lateinit var classes: List<String>
    private var objDetector: ObjectDetector = ObjectDetector()

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classes = readClasses()
        
        // Initialize Ort Session
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, { imageProxy ->
                        performInference(imageProxy)
                    })
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun performInference(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap() ?: return
        val byteArray = bitmapToByteArray(bitmap)
        
        try {
            val result = objDetector.detect(byteArray, ortEnv, ortSession)
            runOnUiThread {
                updateUI(result)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun bitmapToByteArray(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)
        return stream.toByteArray()
    }

    private fun updateUI(result: Result) {
        if (binding.overlay.width <= 0 || binding.overlay.height <= 0) return

        val mutableBitmap = Bitmap.createBitmap(binding.overlay.width, binding.overlay.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 40f
        }

        val backgroundPaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.FILL
        }

        val scaleX = binding.overlay.width.toFloat() / result.outputBitmap.width
        val scaleY = binding.overlay.height.toFloat() / result.outputBitmap.height

        for (box_info in result.outputBox) {
            // box_info: [x_center, y_center, width, height, score, class_id]
            val xCenter = box_info[0] * scaleX
            val yCenter = box_info[1] * scaleY
            val width = box_info[2] * scaleX
            val height = box_info[3] * scaleY

            val left = xCenter - width / 2
            val top = yCenter - height / 2
            val right = xCenter + width / 2
            val bottom = yCenter + height / 2

            canvas.drawRect(left, top, right, bottom, paint)
            val label = "%s:%.2f".format(classes[box_info[5].toInt()], box_info[4])
            val textWidth = textPaint.measureText(label)
            val textHeight = 40f
            val textTop = if (top > textHeight) top else top + textHeight
            canvas.drawRect(left, textTop - textHeight, left + textWidth, textTop, backgroundPaint)
            canvas.drawText(label, left, textTop, textPaint)
        }

        binding.overlay.setImageBitmap(mutableBitmap)
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.yolov8n_with_pre_post_processing
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        ortEnv.close()
        ortSession.close()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    companion object {
        private const val TAG = "ORTObjectDetection"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun ImageProxy.toBitmap(): Bitmap? {
        val image = this.image ?: return null
        val planes = image.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        val matrix = Matrix()
        matrix.postRotate(imageInfo.rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}
