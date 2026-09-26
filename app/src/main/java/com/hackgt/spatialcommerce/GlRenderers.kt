package com.hackgt.spatialcommerce

import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.Matrix
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Pose
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer

class CameraBackgroundRenderer {
    var textureId: Int = -1
        private set

    private var program = 0
    private var positionAttribute = 0
    private var texCoordAttribute = 0
    private var textureUniform = 0
    private val screenCoordinates = floatBufferOf(
        -1f, -1f,
        1f, -1f,
        -1f, 1f,
        1f, 1f,
    )
    private val textureCoordinates = floatBufferOf(
        0f, 1f,
        1f, 1f,
        0f, 0f,
        1f, 0f,
    )

    fun createOnGlThread() {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        textureId = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        program = createProgram(BACKGROUND_VERTEX_SHADER, BACKGROUND_FRAGMENT_SHADER)
        positionAttribute = GLES20.glGetAttribLocation(program, "aPosition")
        texCoordAttribute = GLES20.glGetAttribLocation(program, "aTexCoord")
        textureUniform = GLES20.glGetUniformLocation(program, "uTexture")
    }

    fun draw(frame: Frame) {
        if (frame.hasDisplayGeometryChanged()) {
            screenCoordinates.position(0)
            textureCoordinates.position(0)
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                screenCoordinates,
                Coordinates2d.TEXTURE_NORMALIZED,
                textureCoordinates,
            )
        }

        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glUniform1i(textureUniform, 0)

        screenCoordinates.position(0)
        GLES20.glVertexAttribPointer(positionAttribute, 2, GLES20.GL_FLOAT, false, 0, screenCoordinates)
        GLES20.glEnableVertexAttribArray(positionAttribute)
        textureCoordinates.position(0)
        GLES20.glVertexAttribPointer(texCoordAttribute, 2, GLES20.GL_FLOAT, false, 0, textureCoordinates)
        GLES20.glEnableVertexAttribArray(texCoordAttribute)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(positionAttribute)
        GLES20.glDisableVertexAttribArray(texCoordAttribute)
        GLES20.glDepthMask(true)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    companion object {
        private const val BACKGROUND_VERTEX_SHADER = """
            attribute vec4 aPosition;
            attribute vec2 aTexCoord;
            varying vec2 vTexCoord;
            void main() {
                gl_Position = aPosition;
                vTexCoord = aTexCoord;
            }
        """

        private const val BACKGROUND_FRAGMENT_SHADER = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            uniform samplerExternalOES uTexture;
            varying vec2 vTexCoord;
            void main() {
                gl_FragColor = texture2D(uTexture, vTexCoord);
            }
        """
    }
}

class BoxRenderer {
    private var program = 0
    private var positionAttribute = 0
    private var normalAttribute = 0
    private var mvpUniform = 0
    private var modelUniform = 0
    private var colorUniform = 0

    private val vertices = floatBufferOf(
        -0.5f, -0.5f, 0.5f,  0f, 0f, 1f,
         0.5f, -0.5f, 0.5f,  0f, 0f, 1f,
         0.5f,  0.5f, 0.5f,  0f, 0f, 1f,
        -0.5f,  0.5f, 0.5f,  0f, 0f, 1f,
        -0.5f, -0.5f,-0.5f,  0f, 0f,-1f,
        -0.5f,  0.5f,-0.5f,  0f, 0f,-1f,
         0.5f,  0.5f,-0.5f,  0f, 0f,-1f,
         0.5f, -0.5f,-0.5f,  0f, 0f,-1f,
        -0.5f,  0.5f,-0.5f,  0f, 1f, 0f,
        -0.5f,  0.5f, 0.5f,  0f, 1f, 0f,
         0.5f,  0.5f, 0.5f,  0f, 1f, 0f,
         0.5f,  0.5f,-0.5f,  0f, 1f, 0f,
        -0.5f, -0.5f,-0.5f,  0f,-1f, 0f,
         0.5f, -0.5f,-0.5f,  0f,-1f, 0f,
         0.5f, -0.5f, 0.5f,  0f,-1f, 0f,
        -0.5f, -0.5f, 0.5f,  0f,-1f, 0f,
         0.5f, -0.5f,-0.5f,  1f, 0f, 0f,
         0.5f,  0.5f,-0.5f,  1f, 0f, 0f,
         0.5f,  0.5f, 0.5f,  1f, 0f, 0f,
         0.5f, -0.5f, 0.5f,  1f, 0f, 0f,
        -0.5f, -0.5f,-0.5f, -1f, 0f, 0f,
        -0.5f, -0.5f, 0.5f, -1f, 0f, 0f,
        -0.5f,  0.5f, 0.5f, -1f, 0f, 0f,
        -0.5f,  0.5f,-0.5f, -1f, 0f, 0f,
    )
    private val indices = shortBufferOf(
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
        8, 9, 10, 8, 10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23,
    )
    private val modelMatrix = FloatArray(16)
    private val modelViewMatrix = FloatArray(16)
    private val mvpMatrix = FloatArray(16)

    fun createOnGlThread() {
        program = createProgram(CUBE_VERTEX_SHADER, CUBE_FRAGMENT_SHADER)
        positionAttribute = GLES20.glGetAttribLocation(program, "aPosition")
        normalAttribute = GLES20.glGetAttribLocation(program, "aNormal")
        mvpUniform = GLES20.glGetUniformLocation(program, "uMvp")
        modelUniform = GLES20.glGetUniformLocation(program, "uModel")
        colorUniform = GLES20.glGetUniformLocation(program, "uColor")
    }

    fun drawCube(
        pose: Pose,
        viewMatrix: FloatArray,
        projectionMatrix: FloatArray,
        sizeMeters: Float,
        color: FloatArray,
        liftByHalf: Boolean,
    ) {
        pose.toMatrix(modelMatrix, 0)
        if (liftByHalf) Matrix.translateM(modelMatrix, 0, 0f, sizeMeters / 2f, 0f)
        Matrix.scaleM(modelMatrix, 0, sizeMeters, sizeMeters, sizeMeters)
        drawModel(modelMatrix, viewMatrix, projectionMatrix, color)
    }

    fun drawModel(
        transformedModelMatrix: FloatArray,
        viewMatrix: FloatArray,
        projectionMatrix: FloatArray,
        color: FloatArray,
    ) {
        Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, transformedModelMatrix, 0)
        Matrix.multiplyMM(mvpMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0)

        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(mvpUniform, 1, false, mvpMatrix, 0)
        GLES20.glUniformMatrix4fv(modelUniform, 1, false, transformedModelMatrix, 0)
        GLES20.glUniform4fv(colorUniform, 1, color, 0)

        vertices.position(0)
        GLES20.glVertexAttribPointer(positionAttribute, 3, GLES20.GL_FLOAT, false, 6 * 4, vertices)
        GLES20.glEnableVertexAttribArray(positionAttribute)
        vertices.position(3)
        GLES20.glVertexAttribPointer(normalAttribute, 3, GLES20.GL_FLOAT, false, 6 * 4, vertices)
        GLES20.glEnableVertexAttribArray(normalAttribute)

        indices.position(0)
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, indices.capacity(), GLES20.GL_UNSIGNED_SHORT, indices)
        GLES20.glDisableVertexAttribArray(positionAttribute)
        GLES20.glDisableVertexAttribArray(normalAttribute)
    }

    companion object {
        private const val CUBE_VERTEX_SHADER = """
            uniform mat4 uMvp;
            uniform mat4 uModel;
            attribute vec4 aPosition;
            attribute vec3 aNormal;
            varying vec3 vNormal;
            void main() {
                gl_Position = uMvp * aPosition;
                vNormal = normalize(mat3(uModel) * aNormal);
            }
        """

        private const val CUBE_FRAGMENT_SHADER = """
            precision mediump float;
            uniform vec4 uColor;
            varying vec3 vNormal;
            void main() {
                vec3 light = normalize(vec3(0.3, 1.0, 0.5));
                float brightness = 0.45 + 0.55 * max(dot(normalize(vNormal), light), 0.0);
                gl_FragColor = vec4(uColor.rgb * brightness, uColor.a);
            }
        """
    }
}

class ProductRenderer(private val boxRenderer: BoxRenderer) {
    private val rootMatrix = FloatArray(16)
    private val partMatrix = FloatArray(16)

    fun draw(
        product: ProductPreview,
        anchorPose: Pose,
        yawDegrees: Float,
        viewMatrix: FloatArray,
        projectionMatrix: FloatArray,
    ) {
        check(product.modelAsset == SUPPORTED_MODEL) { "Unsupported local product model: ${product.modelAsset}" }

        anchorPose.toMatrix(rootMatrix, 0)
        Matrix.rotateM(rootMatrix, 0, yawDegrees, 0f, 1f, 0f)

        val width = product.widthMeters
        val depth = product.depthMeters
        val height = product.heightMeters
        val seatTop = height * 0.52f
        val seatThickness = height * 0.10f
        val legHeight = seatTop - seatThickness
        val legWidth = width * 0.09f
        val legDepth = depth * 0.08f
        val legCenterX = (width - legWidth) / 2f
        val frontLegZ = (depth - legDepth) / 2f
        val rearLegZ = -depth * 0.22f

        drawPart(-legCenterX, legHeight / 2f, frontLegZ, legWidth, legHeight, legDepth, WOOD, viewMatrix, projectionMatrix)
        drawPart(legCenterX, legHeight / 2f, frontLegZ, legWidth, legHeight, legDepth, WOOD, viewMatrix, projectionMatrix)
        drawPart(-legCenterX, legHeight / 2f, rearLegZ, legWidth, legHeight, legDepth, WOOD, viewMatrix, projectionMatrix)
        drawPart(legCenterX, legHeight / 2f, rearLegZ, legWidth, legHeight, legDepth, WOOD, viewMatrix, projectionMatrix)

        drawPart(
            x = 0f,
            y = legHeight + seatThickness / 2f,
            z = depth * 0.10f,
            sizeX = width * 0.90f,
            sizeY = seatThickness,
            sizeZ = depth * 0.72f,
            color = CUSHION,
            viewMatrix = viewMatrix,
            projectionMatrix = projectionMatrix,
        )

        val backThickness = depth * 0.10f
        val backHeight = height - seatTop
        drawPart(
            x = 0f,
            y = seatTop + backHeight / 2f,
            z = -depth / 2f + backThickness / 2f,
            sizeX = width * 0.90f,
            sizeY = backHeight,
            sizeZ = backThickness,
            color = CUSHION_DARK,
            viewMatrix = viewMatrix,
            projectionMatrix = projectionMatrix,
        )

        val armWidth = width * 0.07f
        val armHeight = height * 0.07f
        val armCenterX = (width - armWidth) / 2f
        val armCenterY = seatTop + height * 0.18f
        val armDepth = depth * 0.60f
        drawPart(-armCenterX, armCenterY, depth * 0.04f, armWidth, armHeight, armDepth, WOOD_LIGHT, viewMatrix, projectionMatrix)
        drawPart(armCenterX, armCenterY, depth * 0.04f, armWidth, armHeight, armDepth, WOOD_LIGHT, viewMatrix, projectionMatrix)

        val supportHeight = armCenterY - armHeight / 2f - seatTop
        val supportY = seatTop + supportHeight / 2f
        val supportDepth = depth * 0.06f
        val supportZ = depth * 0.26f
        drawPart(-armCenterX, supportY, supportZ, armWidth, supportHeight, supportDepth, WOOD, viewMatrix, projectionMatrix)
        drawPart(armCenterX, supportY, supportZ, armWidth, supportHeight, supportDepth, WOOD, viewMatrix, projectionMatrix)
    }

    private fun drawPart(
        x: Float,
        y: Float,
        z: Float,
        sizeX: Float,
        sizeY: Float,
        sizeZ: Float,
        color: FloatArray,
        viewMatrix: FloatArray,
        projectionMatrix: FloatArray,
    ) {
        System.arraycopy(rootMatrix, 0, partMatrix, 0, 16)
        Matrix.translateM(partMatrix, 0, x, y, z)
        Matrix.scaleM(partMatrix, 0, sizeX, sizeY, sizeZ)
        boxRenderer.drawModel(partMatrix, viewMatrix, projectionMatrix, color)
    }

    companion object {
        private const val SUPPORTED_MODEL = "primitive://lighthouse-lounge-chair-v1"
        private val CUSHION = floatArrayOf(0.08f, 0.52f, 0.58f, 1f)
        private val CUSHION_DARK = floatArrayOf(0.06f, 0.34f, 0.40f, 1f)
        private val WOOD = floatArrayOf(0.34f, 0.16f, 0.07f, 1f)
        private val WOOD_LIGHT = floatArrayOf(0.56f, 0.30f, 0.12f, 1f)
    }
}

private fun createProgram(vertexSource: String, fragmentSource: String): Int {
    val vertexShader = compileShader(GLES20.GL_VERTEX_SHADER, vertexSource)
    val fragmentShader = compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
    return GLES20.glCreateProgram().also { program ->
        GLES20.glAttachShader(program, vertexShader)
        GLES20.glAttachShader(program, fragmentShader)
        GLES20.glLinkProgram(program)
        val status = IntArray(1)
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, status, 0)
        check(status[0] == GLES20.GL_TRUE) { "GL program link failed: ${GLES20.glGetProgramInfoLog(program)}" }
        GLES20.glDeleteShader(vertexShader)
        GLES20.glDeleteShader(fragmentShader)
    }
}

private fun compileShader(type: Int, source: String): Int = GLES20.glCreateShader(type).also { shader ->
    GLES20.glShaderSource(shader, source)
    GLES20.glCompileShader(shader)
    val status = IntArray(1)
    GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
    check(status[0] == GLES20.GL_TRUE) { "GL shader compile failed: ${GLES20.glGetShaderInfoLog(shader)}" }
}

private fun floatBufferOf(vararg values: Float): FloatBuffer =
    ByteBuffer.allocateDirect(values.size * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
        .apply {
            put(values)
            position(0)
        }

private fun shortBufferOf(vararg values: Short): ShortBuffer =
    ByteBuffer.allocateDirect(values.size * 2)
        .order(ByteOrder.nativeOrder())
        .asShortBuffer()
        .apply {
            put(values)
            position(0)
        }
