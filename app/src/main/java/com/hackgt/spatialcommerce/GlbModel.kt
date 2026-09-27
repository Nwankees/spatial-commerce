package com.hackgt.spatialcommerce

import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

class GlbFormatException(message: String) : Exception(message)

/**
 * One triangle mesh decoded from a GLB (glTF 2.0 binary): the subset the backend's
 * normalized preview assets use (float POSITION/NORMAL/TEXCOORD_0, 8/16/32-bit indices,
 * one PBR material with an optional embedded base-color image). Pure Kotlin; no GL here.
 */
class GlbMesh(
    val positions: FloatArray,
    val normals: FloatArray?,
    val texCoords: FloatArray?,
    val indices: IntArray,
    val baseColorImage: ByteArray?,
    val baseColorFactor: FloatArray,
) {
    val vertexCount: Int get() = positions.size / 3

    /** Axis-aligned bounds [minX, minY, minZ, maxX, maxY, maxZ] in mesh units. */
    fun bounds(): FloatArray {
        val b = floatArrayOf(Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE)
        for (i in 0 until vertexCount) {
            for (axis in 0..2) {
                val v = positions[i * 3 + axis]
                if (v < b[axis]) b[axis] = v
                if (v > b[axis + 3]) b[axis + 3] = v
            }
        }
        return b
    }
}

object GlbParser {
    private const val MAGIC = 0x46546C67 // "glTF"
    private const val CHUNK_JSON = 0x4E4F534A
    private const val CHUNK_BIN = 0x004E4942
    const val MAX_VERTICES = 2_000_000

    /** Parses a single-mesh GLB; every malformed input surfaces as [GlbFormatException]. */
    fun parse(bytes: ByteArray): GlbMesh = try {
        parseUnchecked(bytes)
    } catch (e: GlbFormatException) {
        throw e
    } catch (e: org.json.JSONException) {
        throw GlbFormatException("Malformed glTF JSON: ${e.message}")
    } catch (e: IndexOutOfBoundsException) {
        throw GlbFormatException("GLB data is out of range.")
    }

    private fun parseUnchecked(bytes: ByteArray): GlbMesh {
        if (bytes.size < 20) throw GlbFormatException("File too small to be a GLB.")
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        if (buffer.getInt(0) != MAGIC) throw GlbFormatException("Not a GLB file.")
        if (buffer.getInt(4) != 2) throw GlbFormatException("Unsupported glTF version.")
        val totalLength = buffer.getInt(8)
        if (totalLength > bytes.size) throw GlbFormatException("Truncated GLB.")
        var offset = 12
        var json: JSONObject? = null
        var binStart = -1
        var binLength = 0
        while (offset + 8 <= totalLength) {
            val chunkLength = buffer.getInt(offset)
            val chunkType = buffer.getInt(offset + 4)
            val dataStart = offset + 8
            if (chunkLength < 0 || dataStart + chunkLength > totalLength) throw GlbFormatException("Corrupt GLB chunk.")
            when (chunkType) {
                CHUNK_JSON -> json = JSONObject(String(bytes, dataStart, chunkLength, Charsets.UTF_8))
                CHUNK_BIN -> { binStart = dataStart; binLength = chunkLength }
            }
            offset = dataStart + chunkLength
        }
        val gltf = json ?: throw GlbFormatException("GLB has no JSON chunk.")
        if (binStart < 0) throw GlbFormatException("GLB has no binary chunk.")
        gltf.optJSONArray("extensionsRequired")?.let {
            if (it.length() > 0) throw GlbFormatException("GLB requires unsupported extensions: $it")
        }
        val meshes = gltf.optJSONArray("meshes") ?: throw GlbFormatException("GLB has no meshes.")
        if (meshes.length() != 1) throw GlbFormatException("Expected exactly one mesh, found ${meshes.length()}.")
        val primitives = meshes.getJSONObject(0).getJSONArray("primitives")
        if (primitives.length() != 1) throw GlbFormatException("Expected exactly one primitive.")
        val primitive = primitives.getJSONObject(0)
        if (primitive.optInt("mode", 4) != 4) throw GlbFormatException("Only triangle meshes are supported.")
        val reader = AccessorReader(gltf, bytes, binStart, binLength)
        val attributes = primitive.getJSONObject("attributes")
        var positions = reader.floats(attributes.getInt("POSITION"), 3)
        if (positions.size / 3 > MAX_VERTICES) throw GlbFormatException("Mesh is too large for the device.")
        var normals = if (attributes.has("NORMAL")) reader.floats(attributes.getInt("NORMAL"), 3) else null
        val texCoords = if (attributes.has("TEXCOORD_0")) reader.floats(attributes.getInt("TEXCOORD_0"), 2) else null
        val indices = if (primitive.has("indices")) reader.indices(primitive.getInt("indices"))
        else IntArray(positions.size / 3) { it }
        val vertexCount = positions.size / 3
        if (indices.isEmpty() || indices.size % 3 != 0 || indices.any { it < 0 || it >= vertexCount }) {
            throw GlbFormatException("Invalid triangle indices.")
        }
        nodeMatrixFor(gltf)?.let { matrix ->
            positions = transformPoints(positions, matrix, 1f)
            normals = normals?.let { transformPoints(it, matrix, 0f) }
        }
        var image: ByteArray? = null
        var factor = floatArrayOf(1f, 1f, 1f, 1f)
        if (primitive.has("material")) {
            val material = gltf.getJSONArray("materials").getJSONObject(primitive.getInt("material"))
            val pbr = material.optJSONObject("pbrMetallicRoughness")
            pbr?.optJSONArray("baseColorFactor")?.let { f -> factor = FloatArray(4) { f.optDouble(it, 1.0).toFloat() } }
            pbr?.optJSONObject("baseColorTexture")?.let { texture ->
                val source = gltf.getJSONArray("textures").getJSONObject(texture.getInt("index")).getInt("source")
                val imageJson = gltf.getJSONArray("images").getJSONObject(source)
                if (imageJson.has("bufferView")) image = reader.bufferViewBytes(imageJson.getInt("bufferView"))
            }
        }
        return GlbMesh(positions, normals, texCoords, indices, image, factor)
    }

    /** World matrix of the single node that references mesh 0 (column-major), if not identity. */
    private fun nodeMatrixFor(gltf: JSONObject): FloatArray? {
        val nodes = gltf.optJSONArray("nodes") ?: return null
        for (i in 0 until nodes.length()) {
            val node = nodes.getJSONObject(i)
            if (node.optInt("mesh", -1) != 0) continue
            node.optJSONArray("matrix")?.let { m -> return FloatArray(16) { m.getDouble(it).toFloat() } }
            val t = node.optJSONArray("translation")
            val r = node.optJSONArray("rotation")
            val s = node.optJSONArray("scale")
            if (t == null && r == null && s == null) return null
            return trs(t?.toFloats() ?: floatArrayOf(0f, 0f, 0f), r?.toFloats() ?: floatArrayOf(0f, 0f, 0f, 1f), s?.toFloats() ?: floatArrayOf(1f, 1f, 1f))
        }
        return null
    }

    private fun JSONArray.toFloats() = FloatArray(length()) { getDouble(it).toFloat() }

    private fun trs(t: FloatArray, q: FloatArray, s: FloatArray): FloatArray {
        val (x, y, z, w) = q.toList()
        return floatArrayOf(
            (1 - 2 * (y * y + z * z)) * s[0], (2 * (x * y + z * w)) * s[0], (2 * (x * z - y * w)) * s[0], 0f,
            (2 * (x * y - z * w)) * s[1], (1 - 2 * (x * x + z * z)) * s[1], (2 * (y * z + x * w)) * s[1], 0f,
            (2 * (x * z + y * w)) * s[2], (2 * (y * z - x * w)) * s[2], (1 - 2 * (x * x + y * y)) * s[2], 0f,
            t[0], t[1], t[2], 1f,
        )
    }

    private fun transformPoints(values: FloatArray, m: FloatArray, w: Float): FloatArray {
        val out = FloatArray(values.size)
        for (i in 0 until values.size / 3) {
            val x = values[i * 3]; val y = values[i * 3 + 1]; val z = values[i * 3 + 2]
            out[i * 3] = m[0] * x + m[4] * y + m[8] * z + m[12] * w
            out[i * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w
            out[i * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w
        }
        return out
    }

    private class AccessorReader(val gltf: JSONObject, val bytes: ByteArray, val binStart: Int, val binLength: Int) {
        private val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        fun floats(accessorIndex: Int, components: Int): FloatArray {
            val accessor = gltf.getJSONArray("accessors").getJSONObject(accessorIndex)
            if (accessor.getInt("componentType") != 5126) throw GlbFormatException("Only float vertex attributes are supported.")
            if (accessor.optBoolean("normalized", false) || accessor.has("sparse")) throw GlbFormatException("Unsupported accessor.")
            val count = accessor.getInt("count")
            val (start, stride) = locate(accessor, components * 4, count)
            return FloatArray(count * components) { i -> buffer.getFloat(start + (i / components) * stride + (i % components) * 4) }
        }

        fun indices(accessorIndex: Int): IntArray {
            val accessor = gltf.getJSONArray("accessors").getJSONObject(accessorIndex)
            val count = accessor.getInt("count")
            return when (accessor.getInt("componentType")) {
                5121 -> locate(accessor, 1, count).let { (s, st) -> IntArray(count) { bytes[s + it * st].toInt() and 0xFF } }
                5123 -> locate(accessor, 2, count).let { (s, st) -> IntArray(count) { buffer.getShort(s + it * st).toInt() and 0xFFFF } }
                5125 -> locate(accessor, 4, count).let { (s, st) -> IntArray(count) { buffer.getInt(s + it * st) } }
                else -> throw GlbFormatException("Unsupported index type.")
            }
        }

        fun bufferViewBytes(viewIndex: Int): ByteArray {
            val view = gltf.getJSONArray("bufferViews").getJSONObject(viewIndex)
            val offset = view.optInt("byteOffset", 0)
            val length = view.getInt("byteLength")
            if (view.optInt("buffer", 0) != 0 || offset < 0 || offset + length > binLength) throw GlbFormatException("Bad image buffer view.")
            return bytes.copyOfRange(binStart + offset, binStart + offset + length)
        }

        /** Absolute start and stride of an accessor's data, bounds-checked against the BIN chunk. */
        private fun locate(accessor: JSONObject, elementSize: Int, count: Int): Pair<Int, Int> {
            val view = gltf.getJSONArray("bufferViews").getJSONObject(accessor.getInt("bufferView"))
            if (view.optInt("buffer", 0) != 0) throw GlbFormatException("External buffers are not supported.")
            val stride = view.optInt("byteStride", 0).takeIf { it > 0 } ?: elementSize
            val start = view.optInt("byteOffset", 0) + accessor.optInt("byteOffset", 0)
            val end = start + if (count == 0) 0 else (count - 1) * stride + elementSize
            if (count < 0 || start < 0 || end > binLength || end > view.optInt("byteOffset", 0) + view.getInt("byteLength")) {
                throw GlbFormatException("Accessor exceeds its buffer.")
            }
            return (binStart + start) to stride
        }
    }
}
