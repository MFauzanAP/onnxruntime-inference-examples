package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.example.objectdetection.databinding.ActivityMainBinding
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessions: MutableMap<String, OrtSession> = mutableMapOf()
    private lateinit var classes: List<String>
    private var objDetector: ObjectDetector = ObjectDetector()

    private lateinit var cameraExecutor: ExecutorService
    private var faceLandmarker: FaceLandmarker? = null

    // Monitoring State (Matching Python logic)
    private var earConsecFrames = 0
    private var marConsecFrames = 0
    private var headConsecFrames = 0
    private var noFaceFrames = 0
    private var lastFps = 0f
    private var lastTimestamp = System.currentTimeMillis()
    private var frameCount = 0
    
    // Stats for HUD
    private var currentEAR = 0f
    private var currentMAR = 0f
    private var currentHeadPose = "Forward OK"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classes = readClasses()
        initModels()
        initMediaPipe()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun initModels() {
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        
        val modelsToLoad = listOf("sobest", "best")
        for (modelName in modelsToLoad) {
            try {
                val resId = resources.getIdentifier(modelName, "raw", packageName)
                if (resId != 0) {
                    val modelBytes = resources.openRawResource(resId).readBytes()
                    sessions[modelName] = ortEnv.createSession(modelBytes, sessionOptions)
                    Log.i(TAG, "Loaded model: $modelName")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model: $modelName", e)
            }
        }
    }

    private fun initMediaPipe() {
        try {
            val baseOptions = BaseOptions.builder().setModelAssetPath("face_landmarker.task").build()
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinFaceDetectionConfidence(0.7f)
                .setMinTrackingConfidence(0.7f)
                .setRunningMode(RunningMode.IMAGE)
                .build()
            faceLandmarker = FaceLandmarker.createFromOptions(this, options)
        } catch (e: Exception) {
            Log.e(TAG, "MediaPipe init failed. Ensure face_landmarker.task is in assets.", e)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy -> performInference(imageProxy) }
                }
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun performInference(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap() ?: return
        
        try {
            // 1. MediaPipe Landmarks
            val mpImage = com.google.mediapipe.framework.image.BitmapImageBuilder(bitmap).build()
            val mpResult = faceLandmarker?.detect(mpImage)
            val faceFound = (mpResult?.faceLandmarks()?.size ?: 0) > 0

            // 2. Object Detection (Dual YOLO with NMS)
            val allDetections = mutableListOf<DetectionResult>()
            listOf("sobest", "best").forEach { name ->
                sessions[name]?.let { session ->
                    try {
                        val res = objDetector.detect(bitmap, ortEnv, session)
                        res.outputBox.forEach { box ->
                            // Scale coordinates to bitmap pixels immediately based on model dimensions
                            val scaleX = bitmap.width.toFloat() / res.modelWidth
                            val scaleY = bitmap.height.toFloat() / res.modelHeight
                            val scaledBox = floatArrayOf(
                                box[0] * scaleX, box[1] * scaleY, box[2] * scaleX, box[3] * scaleY, box[4], box[5]
                            )
                            allDetections.add(DetectionResult(scaledBox, name))
                        }
                    } catch (e: Exception) { Log.e(TAG, "YOLO $name failed: ${e.message}") }
                }
            }
            val finalDetections = nms(allDetections)

            // 3. Analysis Logic
            val alerts = processAlerts(mpResult, finalDetections, faceFound)

            // Calculate FPS
            frameCount++
            val now = System.currentTimeMillis()
            if (now - lastTimestamp >= 1000) {
                lastFps = frameCount * 1000f / (now - lastTimestamp)
                frameCount = 0
                lastTimestamp = now
            }

            runOnUiThread {
                updateUI(bitmap, mpResult, finalDetections, alerts, lastFps, faceFound)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error", e)
        } finally {
            imageProxy.close()
        }
    }

    private data class DetectionResult(val box: FloatArray, val modelName: String)

    private fun nms(dets: List<DetectionResult>, iouThresh: Float = 0.45f): List<DetectionResult> {
        if (dets.isEmpty()) return emptyList()
        val sorted = dets.sortedByDescending { it.box[4] }
        val selected = mutableListOf<DetectionResult>()
        val active = BooleanArray(sorted.size) { true }
        
        for (i in sorted.indices) {
            if (active[i]) {
                selected.add(sorted[i])
                for (j in i + 1 until sorted.size) {
                    if (active[j] && iou(sorted[i].box, sorted[j].box) > iouThresh) {
                        active[j] = false
                    }
                }
            }
        }
        return selected
    }

    private fun iou(b1: FloatArray, b2: FloatArray): Float {
        // [left, top, right, bottom]
        val x1 = maxOf(b1[0], b2[0]); val y1 = maxOf(b1[1], b2[1])
        val x2 = minOf(b1[2], b2[2]); val y2 = minOf(b1[3], b2[3])
        val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        val a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return if (a1 + a2 - inter > 0) inter / (a1 + a2 - inter) else 0f
    }

    private fun processAlerts(mpResult: FaceLandmarkerResult?, detections: List<DetectionResult>, faceFound: Boolean): List<Pair<String, Int>> {
        val alerts = mutableListOf<Pair<String, Int>>()
        
        mpResult?.faceLandmarks()?.firstOrNull()?.let { landmarks ->
            noFaceFrames = 0
            
            // EAR
            currentEAR = calculateEAR(landmarks)
            if (currentEAR < 0.25f) earConsecFrames++ else earConsecFrames = 0
            if (earConsecFrames >= 5) alerts.add("EYES CLOSED! WAKE UP!" to Color.RED)

            // MAR
            currentMAR = calculateMAR(landmarks)
            if (currentMAR > 0.50f) marConsecFrames++ else marConsecFrames = 0
            if (marConsecFrames >= 7) alerts.add("YAWNING DETECTED!" to Color.rgb(255, 140, 0))
            
            // Pose
            currentHeadPose = classifyHeadDirection(landmarks) ?: "Forward OK"
            if (currentHeadPose != "Forward OK") headConsecFrames++ else headConsecFrames = 0
            if (headConsecFrames >= 20) alerts.add("DISTRACTED: $currentHeadPose" to Color.RED)
        }

        // YOLO Objects
        val labels = detections.map { d -> 
            val id = d.box[5].toInt()
            if (d.modelName == "sobest") classes.getOrElse(id) { "" } else ""
        }
        if (labels.contains("PhoneUse") || labels.contains("Phone")) alerts.add("PUT DOWN PHONE!" to Color.RED)
        if (labels.contains("Smoking") || labels.contains("Cigarette")) alerts.add("NO SMOKING!" to Color.rgb(255, 140, 0))
        
        val hasSeatbelt = labels.contains("Seatbelt")
        if (faceFound && !hasSeatbelt) alerts.add("FASTEN SEATBELT!" to Color.RED)

        if (!faceFound) {
            noFaceFrames++
            if (noFaceFrames > 10) alerts.add("NO FACE DETECTED" to Color.rgb(255, 140, 0))
        }
        
        return alerts
    }

    private fun calculateEAR(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): Float {
        fun dist(i: Int, j: Int) = sqrt((landmarks[i].x() - landmarks[j].x()).pow(2) + (landmarks[i].y() - landmarks[j].y()).pow(2))
        val left = (dist(385, 373) + dist(387, 380)) / (2 * dist(362, 263))
        val right = (dist(160, 153) + dist(158, 144)) / (2 * dist(33, 133))
        return (left + right) / 2f
    }

    private fun calculateMAR(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): Float {
        fun dist(i: Int, j: Int) = sqrt((landmarks[i].x() - landmarks[j].x()).pow(2) + (landmarks[i].y() - landmarks[j].y()).pow(2))
        return (dist(13, 14) + dist(82, 87) + dist(312, 317)) / (3 * dist(78, 308))
    }
    
    private fun classifyHeadDirection(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): String? {
        val nose = landmarks[1]; val leftEye = landmarks[33]; val rightEye = landmarks[263]; val chin = landmarks[152]
        val faceWidth = rightEye.x() - leftEye.x()
        val noseCenterOffset = nose.x() - (leftEye.x() + faceWidth / 2f)
        if (noseCenterOffset < -faceWidth * 0.2f) return "Looking LEFT <"
        if (noseCenterOffset > faceWidth * 0.2f) return "Looking RIGHT >"
        val eyeChinDist = chin.y() - (leftEye.y() + rightEye.y()) / 2f
        if (eyeChinDist < 0.25f) return "Looking UP ^"
        return null
    }

    private fun updateUI(bitmap: Bitmap, mpResult: FaceLandmarkerResult?, detections: List<DetectionResult>, alerts: List<Pair<String, Int>>, fps: Float, faceFound: Boolean) {
        if (binding.overlay.width <= 0 || binding.overlay.height <= 0) return
        
        val overlay = Bitmap.createBitmap(binding.overlay.width, binding.overlay.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(overlay)
        val paint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f }
        val textPaint = Paint().apply { color = Color.WHITE; textSize = 40f; typeface = Typeface.DEFAULT_BOLD }
        val dotPaint = Paint().apply { style = Paint.Style.FILL }
        
        val scaleX = canvas.width.toFloat() / bitmap.width
        val scaleY = canvas.height.toFloat() / bitmap.height

        // 1. Draw Landmarks (Replicating Python landmark dots)
        mpResult?.faceLandmarks()?.firstOrNull()?.let { landmarks ->
            val eyesIndices = listOf(362, 385, 387, 263, 373, 380, 33, 160, 158, 133, 153, 144)
            val mouthIndices = listOf(78, 308, 82, 87, 13, 14, 312, 317)
            
            dotPaint.color = Color.rgb(0, 255, 180) // Eyes (Greenish)
            eyesIndices.forEach { canvas.drawCircle(landmarks[it].x() * canvas.width, landmarks[it].y() * canvas.height, 4f, dotPaint) }
            
            dotPaint.color = Color.rgb(180, 220, 0) // Mouth (Yellowish)
            mouthIndices.forEach { canvas.drawCircle(landmarks[it].x() * canvas.width, landmarks[it].y() * canvas.height, 4f, dotPaint) }
        }

        // 2. Draw YOLO Boxes
        detections.forEach { d ->
            paint.color = if (d.modelName == "sobest") Color.RED else Color.GREEN
            val left = d.box[0] * scaleX; val top = d.box[1] * scaleY
            val right = d.box[2] * scaleX; val bottom = d.box[3] * scaleY
            canvas.drawRect(left, top, right, bottom, paint)
            
            val label = if (d.modelName == "sobest") classes.getOrElse(d.box[5].toInt()) { "Obj" } else "Best"
            canvas.drawText("%s %.2f".format(label, d.box[4]), left, top - 10, textPaint)
        }

        // 3. Draw HUD (Top Left)
        canvas.drawRect(0f, 0f, 450f, 220f, Paint().apply { color = Color.BLACK; alpha = 160 })
        val statusText = if (faceFound) "Face: MP OK" else "Face: NOT FOUND"
        val statusColor = if (faceFound) Color.GREEN else Color.RED
        textPaint.color = statusColor; textPaint.textSize = 35f
        canvas.drawText("FPS: %.1f | %s".format(fps, statusText), 20f, 45f, textPaint)
        
        textPaint.color = if (currentEAR < 0.25f) Color.RED else Color.GREEN
        canvas.drawText("EAR: %.3f (thr 0.25)".format(currentEAR), 20f, 90f, textPaint)
        
        textPaint.color = if (currentMAR > 0.50f) Color.rgb(255, 140, 0) else Color.GREEN
        canvas.drawText("MAR: %.3f (thr 0.50)".format(currentMAR), 20f, 135f, textPaint)
        
        textPaint.color = Color.YELLOW
        canvas.drawText("YOLO Objs: ${detections.size}", 20f, 180f, textPaint)

        // 4. Top Right Pose & Seatbelt
        val poseColor = if (currentHeadPose == "Forward OK") Color.GREEN else Color.RED
        val poseTextWidth = textPaint.measureText(currentHeadPose)
        canvas.drawText(currentHeadPose, canvas.width - poseTextWidth - 20f, 45f, textPaint.apply { color = poseColor })
        
        val hasSeatbelt = detections.any { d -> d.modelName == "sobest" && classes.getOrElse(d.box[5].toInt()){""} == "Seatbelt" }
        val sbText = if (hasSeatbelt) "SEATBELT ON" else "NO SEATBELT"
        val sbColor = if (hasSeatbelt) Color.GREEN else Color.RED
        val sbTextWidth = textPaint.measureText(sbText)
        canvas.drawText(sbText, canvas.width - sbTextWidth - 20f, 100f, textPaint.apply { color = sbColor })

        // 5. Draw Alerts (Bottom Centered)
        var ay = canvas.height - 80f
        alerts.reversed().forEach { (msg, col) ->
            textPaint.color = col; textPaint.textSize = 50f
            val tw = textPaint.measureText(msg)
            canvas.drawRect(canvas.width/2f - tw/2 - 10, ay - 55, canvas.width/2f + tw/2 + 10, ay + 10, Paint().apply { color = Color.BLACK; alpha = 180 })
            canvas.drawText(msg, canvas.width/2f - tw/2, ay, textPaint)
            ay -= 70f
        }

        binding.overlay.setImageBitmap(overlay)
    }

    private fun readClasses() = resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all { ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        ortEnv.close()
        sessions.values.forEach { it.close() }
        faceLandmarker?.close()
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun ImageProxy.toBitmap(): Bitmap? {
        val image = this.image ?: return null
        val planes = image.planes
        val yBuffer = planes[0].buffer; val uBuffer = planes[1].buffer; val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining(); val uSize = uBuffer.remaining(); val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize); vBuffer.get(nv21, ySize, vSize); uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
        
        val matrix = Matrix().apply { 
            postRotate(imageInfo.rotationDegrees.toFloat())
            // Horizontal flip for front camera to match user expectation (mirror)
            postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    companion object {
        private const val TAG = "ORTObjectDetection"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
