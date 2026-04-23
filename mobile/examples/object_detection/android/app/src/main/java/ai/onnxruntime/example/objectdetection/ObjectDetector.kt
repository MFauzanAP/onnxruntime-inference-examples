package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.*
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.*

internal data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>,
    var modelWidth: Int = 640,
    var modelHeight: Int = 640
) {}

internal class ObjectDetector(
) {
    fun detect(bitmap: Bitmap, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        val inputName = ortSession.inputNames.iterator().next()
        val inputInfo = ortSession.inputInfo[inputName]
        val nodeInfo = inputInfo?.info
        val type = if (nodeInfo is TensorInfo) nodeInfo.type else null
        val shape = if (nodeInfo is TensorInfo) nodeInfo.shape else longArrayOf()

        return if (type == OnnxJavaType.FLOAT) {
            // Standard model: expects Float tensor [1, 3, H, W]
            val imgHeight = if (shape.size > 2 && shape[2] > 0) shape[2].toInt() else 640
            val imgWidth = if (shape.size > 3 && shape[3] > 0) shape[3].toInt() else 640
            
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, imgWidth, imgHeight, true)
            val floatBuffer = bitmapToFloatBuffer(resizedBitmap)
            
            val tensorShape = longArrayOf(1, 3, imgHeight.toLong(), imgWidth.toLong())
            val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, tensorShape)
            runInference(inputTensor, ortSession, imgWidth, imgHeight)
        } else {
            // Pre-processing model: expects UINT8 JPEG bytes
            val stream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream)
            detect(stream.toByteArray(), ortEnv, ortSession)
        }
    }

    fun detect(rawImageBytes: ByteArray, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        val shape = longArrayOf(rawImageBytes.size.toLong())
        val inputName = ortSession.inputNames.iterator().next()
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            ByteBuffer.wrap(rawImageBytes),
            shape,
            OnnxJavaType.UINT8
        )
        return runInference(inputTensor, ortSession, 640, 640)
    }

    private fun runInference(inputTensor: OnnxTensor, ortSession: OrtSession, mWidth: Int, mHeight: Int): Result {
        inputTensor.use {
            val inputName = ortSession.inputNames.iterator().next()
            val outputNames = ortSession.outputNames.toList()
            val output = ortSession.run(Collections.singletonMap(inputName, inputTensor))

            output.use {
                val CONF_THRESHOLD = 0.3f

                // YuNet Face Detection
                if (ortSession.outputInfo.size == 1 && !outputNames.contains("image_out")) {
                    val rawOutput = output.get(0).value as Array<Array<FloatArray>>
                    val faces = rawOutput[0]
                    val filteredBoxes = faces.filter { it[14] >= CONF_THRESHOLD }.map { face ->
                        floatArrayOf(face[0], face[1], face[0] + face[2], face[1] + face[3], face[14], 0f)
                    }.toTypedArray()
                    return Result(Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888), filteredBoxes, mWidth, mHeight)
                }

                // YOLO
                val rawOutput = if (outputNames.contains("image_out")) {
                    (output.get("image_out").get().value) as? ByteArray
                } else {
                    output.get(0).value as? ByteArray
                }

                val boxOutput = if (outputNames.contains("scaled_box_out_next")) {
                    (output.get("scaled_box_out_next").get().value) as? Array<FloatArray>
                } else if (output.size() > 1) {
                    output.get(1).value as? Array<FloatArray>
                } else {
                    null
                }
                
                val filteredBoxOutput = boxOutput?.filter { it[4] >= CONF_THRESHOLD }?.toTypedArray() ?: emptyArray()

                val outputImageBitmap = try {
                    if (rawOutput != null) byteArrayToBitmap(rawOutput) else Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
                } catch (e: Exception) {
                    Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
                }

                return Result(outputImageBitmap, filteredBoxOutput, mWidth, mHeight)
            }
        }
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val floatBuffer = FloatBuffer.allocate(1 * 3 * bitmap.width * bitmap.height)
        floatBuffer.rewind()

        val area = bitmap.width * bitmap.height
        val bitmapPixels = IntArray(area)
        bitmap.getPixels(bitmapPixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // BGR planed format (standard for many ONNX models including YuNet)
        for (i in 0 until area) {
            val pixel = bitmapPixels[i]
            floatBuffer.put(i, (pixel and 0xFF) / 255.0f)                 // Blue
            floatBuffer.put(i + area, ((pixel shr 8) and 0xFF) / 255.0f)   // Green
            floatBuffer.put(i + area * 2, ((pixel shr 16) and 0xFF) / 255.0f) // Red
        }

        floatBuffer.rewind()
        return floatBuffer
    }

    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        return detect(inputStream.readBytes(), ortEnv, ortSession)
    }

    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }
}