package com.hackgt.spatialcommerce

import android.graphics.BitmapFactory
import android.opengl.GLES20
import android.opengl.GLUtils
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Draws one decoded GLB mesh with its base-color texture using the app's existing
 * OpenGL ES 2.0 context (no rendering framework). Create/draw/release on the GL thread.
 */
class TexturedMeshRenderer {
    private var program = 0
    private var positionAttribute = 0
    private var normalAttribute = 0
    private var uvAttribute = 0
    private var mvpUniform = 0
    private var modelUniform = 0
    private var colorUniform = 0
    private var textureUniform = 0
    private var hasTextureUniform = 0
    private val buffers = IntArray(4) // positions, normals, uvs, indices
    private var texture = 0
    private var indexCount = 0
    private var indexType = GLES20.GL_UNSIGNED_SHORT
    private var hasNormals = false
    private var hasUvs = false
    private var baseColor = floatArrayOf(1f, 1f, 1f, 1f)
    private val modelView = FloatArray(16)
    private val mvp = FloatArray(16)

    val isLoaded: Boolean get() = indexCount > 0

    /** Uploads [mesh]; throws IllegalStateException if the device cannot draw it. */
    fun upload(mesh: GlbMesh) {
        release()
        if (program == 0) {
            program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
            positionAttribute = GLES20.glGetAttribLocation(program, "aPosition")
            normalAttribute = GLES20.glGetAttribLocation(program, "aNormal")
            uvAttribute = GLES20.glGetAttribLocation(program, "aUv")
            mvpUniform = GLES20.glGetUniformLocation(program, "uMvp")
            modelUniform = GLES20.glGetUniformLocation(program, "uModel")
            colorUniform = GLES20.glGetUniformLocation(program, "uColor")
            textureUniform = GLES20.glGetUniformLocation(program, "uTexture")
            hasTextureUniform = GLES20.glGetUniformLocation(program, "uHasTexture")
        }
        val needsUint = mesh.vertexCount > 65535
        if (needsUint && !GLES20.glGetString(GLES20.GL_EXTENSIONS).orEmpty().contains("GL_OES_element_index_uint")) {
            throw IllegalStateException("Model has ${mesh.vertexCount} vertices; this device cannot draw 32-bit indices.")
        }
        GLES20.glGenBuffers(4, buffers, 0)
        uploadFloats(buffers[0], mesh.positions)
        hasNormals = mesh.normals != null
        mesh.normals?.let { uploadFloats(buffers[1], it) }
        hasUvs = mesh.texCoords != null
        mesh.texCoords?.let { uploadFloats(buffers[2], it) }
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, buffers[3])
        if (needsUint) {
            val data = ByteBuffer.allocateDirect(mesh.indices.size * 4).order(ByteOrder.nativeOrder()).asIntBuffer()
            data.put(mesh.indices).position(0)
            GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size * 4, data, GLES20.GL_STATIC_DRAW)
            indexType = GLES20.GL_UNSIGNED_INT
        } else {
            val data = ByteBuffer.allocateDirect(mesh.indices.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
            mesh.indices.forEach { data.put(it.toShort()) }
            data.position(0)
            GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size * 2, data, GLES20.GL_STATIC_DRAW)
            indexType = GLES20.GL_UNSIGNED_SHORT
        }
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        baseColor = mesh.baseColorFactor
        texture = mesh.baseColorImage?.let { bytes ->
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return@let 0
            val ids = IntArray(1)
            GLES20.glGenTextures(1, ids, 0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, ids[0])
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR_MIPMAP_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
            GLES20.glGenerateMipmap(GLES20.GL_TEXTURE_2D)
            bitmap.recycle()
            ids[0]
        } ?: 0
        indexCount = mesh.indices.size
        val error = GLES20.glGetError()
        if (error != GLES20.GL_NO_ERROR) {
            release()
            throw IllegalStateException("OpenGL error $error while uploading the model.")
        }
    }

    fun draw(modelMatrix: FloatArray, viewMatrix: FloatArray, projectionMatrix: FloatArray) {
        if (!isLoaded) return
        Matrix.multiplyMM(modelView, 0, viewMatrix, 0, modelMatrix, 0)
        Matrix.multiplyMM(mvp, 0, projectionMatrix, 0, modelView, 0)
        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(mvpUniform, 1, false, mvp, 0)
        GLES20.glUniformMatrix4fv(modelUniform, 1, false, modelMatrix, 0)
        GLES20.glUniform4fv(colorUniform, 1, baseColor, 0)
        GLES20.glUniform1i(hasTextureUniform, if (texture != 0 && hasUvs) 1 else 0)
        if (texture != 0) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture)
            GLES20.glUniform1i(textureUniform, 0)
        }
        bindAttribute(buffers[0], positionAttribute, 3)
        if (hasNormals) bindAttribute(buffers[1], normalAttribute, 3) else GLES20.glVertexAttrib3f(normalAttribute, 0f, 1f, 0f)
        if (hasUvs) bindAttribute(buffers[2], uvAttribute, 2) else GLES20.glVertexAttrib2f(uvAttribute, 0f, 0f)
        // Reconstructed meshes can have inconsistent winding: draw both sides.
        GLES20.glDisable(GLES20.GL_CULL_FACE)
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, buffers[3])
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, indexCount, indexType, 0)
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
        GLES20.glDisableVertexAttribArray(positionAttribute)
        GLES20.glDisableVertexAttribArray(normalAttribute)
        GLES20.glDisableVertexAttribArray(uvAttribute)
    }

    /** The GL context was recreated: old handles are invalid, forget them without GL calls. */
    fun invalidate() {
        program = 0
        buffers.fill(0)
        texture = 0
        indexCount = 0
    }

    fun release() {
        if (buffers[0] != 0) GLES20.glDeleteBuffers(4, buffers, 0)
        buffers.fill(0)
        if (texture != 0) GLES20.glDeleteTextures(1, intArrayOf(texture), 0)
        texture = 0
        indexCount = 0
    }

    private fun uploadFloats(buffer: Int, values: FloatArray) {
        val data = ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        data.put(values).position(0)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, buffer)
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, values.size * 4, data, GLES20.GL_STATIC_DRAW)
    }

    private fun bindAttribute(buffer: Int, attribute: Int, size: Int) {
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, buffer)
        GLES20.glVertexAttribPointer(attribute, size, GLES20.GL_FLOAT, false, 0, 0)
        GLES20.glEnableVertexAttribArray(attribute)
    }

    companion object {
        private const val VERTEX_SHADER = """
            uniform mat4 uMvp;
            uniform mat4 uModel;
            attribute vec4 aPosition;
            attribute vec3 aNormal;
            attribute vec2 aUv;
            varying vec3 vNormal;
            varying vec2 vUv;
            void main() {
                gl_Position = uMvp * aPosition;
                vNormal = normalize(mat3(uModel) * aNormal);
                vUv = aUv;
            }
        """

        private const val FRAGMENT_SHADER = """
            precision mediump float;
            uniform vec4 uColor;
            uniform sampler2D uTexture;
            uniform int uHasTexture;
            varying vec3 vNormal;
            varying vec2 vUv;
            void main() {
                vec4 base = uColor;
                if (uHasTexture == 1) base *= texture2D(uTexture, vUv);
                vec3 light = normalize(vec3(0.3, 1.0, 0.5));
                float diffuse = abs(dot(normalize(vNormal), light));
                gl_FragColor = vec4(base.rgb * (0.6 + 0.4 * diffuse), 1.0);
            }
        """
    }
}
