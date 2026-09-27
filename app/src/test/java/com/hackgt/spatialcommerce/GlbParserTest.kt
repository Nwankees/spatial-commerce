package com.hackgt.spatialcommerce

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class GlbParserTest {
    /** Builds a one-triangle GLB: positions, uv, uint16 indices, optional node transform and embedded image. */
    private fun glb(
        node: JSONObject = JSONObject().put("mesh", 0),
        withImage: Boolean = true,
        mutate: (JSONObject) -> Unit = {},
        indices: ShortArray = shortArrayOf(0, 1, 2),
    ): ByteArray {
        val positions = floatArrayOf(0f, 0f, 0f, 1f, 0f, 0f, 0f, 2f, 0f)
        val uvs = floatArrayOf(0f, 0f, 1f, 0f, 0f, 1f)
        val image = byteArrayOf(1, 2, 3, 4, 5)
        val bin = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN)
        positions.forEach { bin.putFloat(it) } // 0..36
        uvs.forEach { bin.putFloat(it) } // 36..60
        indices.forEach { bin.putShort(it) } // 60..66
        bin.position(68)
        bin.put(image) // 68..73
        val binLength = (73 + 3) / 4 * 4
        val views = JSONArray()
            .put(JSONObject().put("buffer", 0).put("byteOffset", 0).put("byteLength", 36))
            .put(JSONObject().put("buffer", 0).put("byteOffset", 36).put("byteLength", 24))
            .put(JSONObject().put("buffer", 0).put("byteOffset", 60).put("byteLength", indices.size * 2))
            .put(JSONObject().put("buffer", 0).put("byteOffset", 68).put("byteLength", 5))
        val accessors = JSONArray()
            .put(JSONObject().put("bufferView", 0).put("componentType", 5126).put("count", 3).put("type", "VEC3"))
            .put(JSONObject().put("bufferView", 1).put("componentType", 5126).put("count", 3).put("type", "VEC2"))
            .put(JSONObject().put("bufferView", 2).put("componentType", 5123).put("count", indices.size).put("type", "SCALAR"))
        val primitive = JSONObject()
            .put("attributes", JSONObject().put("POSITION", 0).put("TEXCOORD_0", 1))
            .put("indices", 2)
        val gltf = JSONObject()
            .put("asset", JSONObject().put("version", "2.0"))
            .put("buffers", JSONArray().put(JSONObject().put("byteLength", binLength)))
            .put("bufferViews", views)
            .put("accessors", accessors)
            .put("nodes", JSONArray().put(node))
            .put("meshes", JSONArray().put(JSONObject().put("primitives", JSONArray().put(primitive))))
        if (withImage) {
            primitive.put("material", 0)
            gltf.put("materials", JSONArray().put(JSONObject().put("pbrMetallicRoughness",
                JSONObject().put("baseColorTexture", JSONObject().put("index", 0)).put("baseColorFactor", JSONArray(listOf(1, 0.5, 1, 1))))))
            gltf.put("textures", JSONArray().put(JSONObject().put("source", 0)))
            gltf.put("images", JSONArray().put(JSONObject().put("bufferView", 3).put("mimeType", "image/png")))
        }
        mutate(gltf)
        var jsonBytes = gltf.toString().toByteArray(Charsets.UTF_8)
        val pad = (4 - jsonBytes.size % 4) % 4
        jsonBytes += ByteArray(pad) { ' '.code.toByte() }
        val total = 12 + 8 + jsonBytes.size + 8 + binLength
        val out = ByteBuffer.allocate(total).order(ByteOrder.LITTLE_ENDIAN)
        out.putInt(0x46546C67).putInt(2).putInt(total)
        out.putInt(jsonBytes.size).putInt(0x4E4F534A).put(jsonBytes)
        out.putInt(binLength).putInt(0x004E4942).put(bin.array(), 0, binLength)
        return out.array()
    }

    private fun assertRejected(bytes: ByteArray) {
        try {
            GlbParser.parse(bytes)
            fail("expected GlbFormatException")
        } catch (_: GlbFormatException) {
        }
    }

    @Test
    fun parsesTexturedTriangle() {
        val mesh = GlbParser.parse(glb())
        assertEquals(3, mesh.vertexCount)
        assertArrayEquals(intArrayOf(0, 1, 2), mesh.indices)
        assertNotNull(mesh.texCoords)
        assertArrayEquals(byteArrayOf(1, 2, 3, 4, 5), mesh.baseColorImage)
        assertEquals(0.5f, mesh.baseColorFactor[1], 1e-6f)
        val b = mesh.bounds()
        assertArrayEquals(floatArrayOf(0f, 0f, 0f, 1f, 2f, 0f), b, 1e-6f)
    }

    @Test
    fun appliesNodeTranslationAndScale() {
        val node = JSONObject().put("mesh", 0)
            .put("translation", JSONArray(listOf(1, 0, 0)))
            .put("scale", JSONArray(listOf(2, 2, 2)))
        val b = GlbParser.parse(glb(node = node)).bounds()
        assertArrayEquals(floatArrayOf(1f, 0f, 0f, 3f, 4f, 0f), b, 1e-6f)
    }

    @Test
    fun rejectsGarbageAndTruncation() {
        assertRejected(ByteArray(10))
        assertRejected(ByteArray(64) { 7 })
        val good = glb()
        assertRejected(good.copyOf(good.size - 20))
    }

    @Test
    fun rejectsRequiredExtensionsAndMultipleMeshes() {
        assertRejected(glb(mutate = { it.put("extensionsRequired", JSONArray().put("KHR_draco_mesh_compression")) }))
        assertRejected(glb(mutate = { g -> g.getJSONArray("meshes").put(g.getJSONArray("meshes").getJSONObject(0)) }))
    }

    @Test
    fun rejectsOutOfRangeIndicesAndAccessors() {
        assertRejected(glb(indices = shortArrayOf(0, 1, 9)))
        assertRejected(glb(mutate = { g -> g.getJSONArray("accessors").getJSONObject(0).put("count", 100) }))
        assertRejected(glb(mutate = { g -> g.getJSONArray("accessors").getJSONObject(0).put("componentType", 5123) }))
        assertRejected(glb(mutate = { g -> g.getJSONArray("meshes").getJSONObject(0).getJSONArray("primitives").getJSONObject(0).remove("attributes") }))
    }

    @Test
    fun untexturedMeshHasNoImage() {
        val mesh = GlbParser.parse(glb(withImage = false))
        assertTrue(mesh.baseColorImage == null)
    }
}
