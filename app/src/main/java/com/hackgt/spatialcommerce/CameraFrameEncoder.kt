package com.hackgt.spatialcommerce

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream

data class CapturedCameraFrame(
    val jpegBytes: ByteArray,
    val rotationDegrees: Int,
)

object CameraFrameEncoder {
    fun toJpeg(image: Image, quality: Int = 88): ByteArray {
        require(image.format == ImageFormat.YUV_420_888) {
            "Expected an ARCore YUV_420_888 camera image."
        }
        val width = image.width
        val height = image.height
        require(width % 2 == 0 && height % 2 == 0) {
            "Camera image dimensions must be even."
        }

        val planes = image.planes
        require(planes.size == 3) { "Expected three YUV camera planes." }
        val nv21 = ByteArray(width * height + width * height / 2)

        copyLumaPlane(planes[0], width, height, nv21)
        copyChromaPlanes(
            uPlane = planes[1],
            vPlane = planes[2],
            width = width,
            height = height,
            output = nv21,
        )

        val stream = ByteArrayOutputStream()
        val compressed = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            .compressToJpeg(Rect(0, 0, width, height), quality, stream)
        check(compressed) { "Could not encode the ARCore camera image." }
        return stream.toByteArray()
    }

    private fun copyLumaPlane(
        plane: Image.Plane,
        width: Int,
        height: Int,
        output: ByteArray,
    ) {
        val buffer = plane.buffer
        val base = buffer.position()
        for (row in 0 until height) {
            val rowStart = base + row * plane.rowStride
            for (column in 0 until width) {
                output[row * width + column] = buffer.get(rowStart + column * plane.pixelStride)
            }
        }
    }

    private fun copyChromaPlanes(
        uPlane: Image.Plane,
        vPlane: Image.Plane,
        width: Int,
        height: Int,
        output: ByteArray,
    ) {
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val uBase = uBuffer.position()
        val vBase = vBuffer.position()
        var outputIndex = width * height
        for (row in 0 until height / 2) {
            val uRowStart = uBase + row * uPlane.rowStride
            val vRowStart = vBase + row * vPlane.rowStride
            for (column in 0 until width / 2) {
                output[outputIndex++] = vBuffer.get(vRowStart + column * vPlane.pixelStride)
                output[outputIndex++] = uBuffer.get(uRowStart + column * uPlane.pixelStride)
            }
        }
    }
}
