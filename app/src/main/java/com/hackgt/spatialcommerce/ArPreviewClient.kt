package com.hackgt.spatialcommerce

import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

data class ArAssetStatus(
    val assetId: String?,
    /** ready | generating | unavailable | failed */
    val status: String,
    val assetUrl: String?,
    val scale: ArScale?,
    val normalizedSize: FloatArray?,
    val cached: Boolean,
    val previewLabel: String,
    val message: String?,
    val retryable: Boolean,
    val totalGenerationSeconds: Double?,
    val glbBytes: Int?,
    val timings: Map<String, Double>,
    val cacheOutcome: String?,
    val dimensionLookupPath: String?,
    val dimensionLookupCacheHit: Boolean?,
)

data class LoadedArAsset(
    val status: ArAssetStatus,
    val mesh: GlbMesh,
    val preparationMillis: Long,
    val downloadMillis: Long,
    val parseMillis: Long,
)

/**
 * Requests a real-product AR preview from the backend, polls while it is generated,
 * then downloads and parses the normalized GLB — all off the UI thread.
 */
class ArPreviewClient(private val baseUrl: String = "http://127.0.0.1:8000") {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    @Volatile private var generation = 0

    fun cancel() { generation += 1 }

    fun close() { executor.shutdownNow() }

    fun request(
        product: ProductCandidate,
        onProgress: (ArAssetStatus) -> Unit,
        callback: (Result<LoadedArAsset>) -> Unit,
    ) {
        val myGeneration = ++generation
        executor.execute {
            callback(runCatching {
                val requestStarted = System.currentTimeMillis()
                var status = parseStatus(post("/api/v1/products/ar-preview", JSONObject().apply {
                    put("productId", product.id)
                    put("productUrl", product.productUrl)
                }))
                val deadline = System.currentTimeMillis() + MAX_WAIT_MS
                while (status.status == "generating") {
                    if (myGeneration != generation) throw ArPreviewException("Cancelled.")
                    onProgress(status)
                    if (System.currentTimeMillis() > deadline) throw ArPreviewException("3D preview generation is taking too long. Please retry.")
                    Thread.sleep(POLL_MS)
                    status = parseStatus(get("/api/v1/ar-assets/${status.assetId}"))
                }
                when (status.status) {
                    "ready" -> {
                        val url = status.assetUrl ?: throw ArPreviewException("The preview has no model URL.")
                        val preparationMillis = System.currentTimeMillis() - requestStarted
                        val t0 = System.currentTimeMillis()
                        val bytes = download(url)
                        val t1 = System.currentTimeMillis()
                        val mesh = try { GlbParser.parse(bytes) } catch (e: GlbFormatException) {
                            throw ArPreviewException("The 3D model could not be loaded: ${e.message}")
                        }
                        LoadedArAsset(status, mesh, preparationMillis, t1 - t0, System.currentTimeMillis() - t1)
                    }
                    else -> throw ArPreviewException(status.message ?: "3D preview unavailable.", status)
                }
            })
        }
    }

    private fun parseStatus(json: JSONObject): ArAssetStatus {
        val scaleJson = json.optJSONObject("scale")
        val scale = scaleJson?.let {
            ArScale(
                widthMeters = it.getDouble("widthMeters"), depthMeters = it.getDouble("depthMeters"),
                heightMeters = it.getDouble("heightMeters"), scaleX = it.getDouble("scaleX").toFloat(),
                scaleY = it.getDouble("scaleY").toFloat(), scaleZ = it.getDouble("scaleZ").toFloat(),
                axesSwapped = it.optBoolean("axesSwapped", false), maxAxisDistortion = it.optDouble("maxAxisDistortion", 0.0),
            )
        }
        val size = json.optJSONObject("normalizedBounds")?.optJSONArray("size")?.let { a -> FloatArray(3) { a.getDouble(it).toFloat() } }
        val timings = json.optJSONObject("timings")
        val timingValues = buildMap {
            val keys = timings?.keys()
            while (keys != null && keys.hasNext()) {
                val key = keys.next()
                timings.optDouble(key).takeIf { !it.isNaN() }?.let { put(key, it) }
            }
        }
        val dimensionSource = json.optJSONObject("dimensionSource")
        return ArAssetStatus(
            assetId = json.optString("assetId").takeIf { it.isNotBlank() && it != "null" },
            status = json.getString("status"),
            assetUrl = json.optString("assetUrl").takeIf { it.startsWith("/api/v1/ar-assets/") },
            scale = scale, normalizedSize = size, cached = json.optBoolean("cached", false),
            previewLabel = json.optString("previewLabel", "AI-generated 3D preview scaled to verified product dimensions."),
            message = json.optString("message").takeIf { it.isNotBlank() && it != "null" },
            retryable = json.optBoolean("retryable", false),
            totalGenerationSeconds = timings?.optDouble("totalGenerationSeconds")?.takeIf { !it.isNaN() },
            glbBytes = json.optInt("glbBytes", -1).takeIf { it >= 0 },
            timings = timingValues,
            cacheOutcome = json.optString("cacheOutcome").takeIf { it in setOf("hit", "miss", "joined") },
            dimensionLookupPath = dimensionSource?.optString("lookupPath")?.takeIf { it.isNotBlank() && it != "null" },
            dimensionLookupCacheHit = dimensionSource?.takeIf { it.has("lookupCacheHit") }?.optBoolean("lookupCacheHit"),
        )
    }

    private fun post(path: String, body: JSONObject): JSONObject = call(path, "POST", body.toString())

    private fun get(path: String): JSONObject = call(path, "GET", null)

    private fun call(path: String, method: String, body: String?): JSONObject {
        val connection = open(path, READ_TIMEOUT_MS).apply {
            requestMethod = method
            setRequestProperty("Accept", "application/json")
            if (body != null) {
                doOutput = true
                setRequestProperty("Content-Type", "application/json; charset=utf-8")
            }
        }
        try {
            body?.let { b -> connection.outputStream.bufferedWriter(Charsets.UTF_8).use { it.write(b) } }
            val code = connection.responseCode
            val text = (if (code in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader(Charsets.UTF_8)?.use { it.readText().take(200_000) }.orEmpty()
            if (code !in 200..299) {
                val detail = runCatching { JSONObject(text).optString("detail") }.getOrNull()
                throw ArPreviewException(detail?.takeIf { it.isNotBlank() } ?: "3D preview request failed (HTTP $code).")
            }
            return JSONObject(text)
        } catch (e: ArPreviewException) {
            throw e
        } catch (e: SocketTimeoutException) {
            throw ArPreviewException("The 3D preview request timed out. Please retry.")
        } catch (e: IOException) {
            throw ArPreviewException("Cannot reach the local backend. Start it, check adb reverse, and retry.")
        } finally {
            connection.disconnect()
        }
    }

    private fun download(path: String): ByteArray {
        val connection = open(path, DOWNLOAD_TIMEOUT_MS)
        try {
            if (connection.responseCode != 200) throw ArPreviewException("Model download failed (HTTP ${connection.responseCode}).")
            val out = ByteArrayOutputStream()
            connection.inputStream.use { input ->
                val chunk = ByteArray(64 * 1024)
                while (true) {
                    val n = input.read(chunk)
                    if (n < 0) break
                    out.write(chunk, 0, n)
                    if (out.size() > MAX_MODEL_BYTES) throw ArPreviewException("The 3D model is too large to load.")
                }
            }
            return out.toByteArray()
        } catch (e: ArPreviewException) {
            throw e
        } catch (e: IOException) {
            throw ArPreviewException("The 3D model download failed. Please retry.")
        } finally {
            connection.disconnect()
        }
    }

    private fun open(path: String, timeoutMs: Int) = (URL("$baseUrl$path").openConnection() as HttpURLConnection).apply {
        connectTimeout = 5_000
        readTimeout = timeoutMs
    }

    class ArPreviewException(message: String, val status: ArAssetStatus? = null) : IOException(message)

    companion object {
        private const val POLL_MS = 2_000L
        private const val MAX_WAIT_MS = 6 * 60_000L
        private const val READ_TIMEOUT_MS = 30_000
        private const val DOWNLOAD_TIMEOUT_MS = 60_000
        private const val MAX_MODEL_BYTES = 60 * 1024 * 1024
    }
}
