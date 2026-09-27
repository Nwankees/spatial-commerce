package com.hackgt.spatialcommerce

import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Talks to the backend's product-search endpoint. The provider (SerpApi) and its
 * key live only on the backend; the app never contacts a shopping provider directly.
 */
class ProductSearchClient(
    private val baseUrl: String = "http://127.0.0.1:8000",
) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()

    fun search(
        analysis: VisualProductAnalysis,
        frame: CapturedCameraFrame,
        callback: (Result<ProductSearchResult>) -> Unit,
    ) {
        executor.execute {
            callback(runCatching { performRequest(analysis, frame) })
        }
    }

    /**
     * Asks the backend for explicit dimensions of a product it previously returned.
     * Only identifiers are sent; the backend never trusts client-supplied dimensions.
     */
    fun resolveDimensions(
        product: ProductCandidate,
        callback: (Result<ResolvedDimensions>) -> Unit,
    ) {
        executor.execute {
            callback(runCatching { performDimensionRequest(product) })
        }
    }

    fun close() {
        executor.shutdownNow()
    }

    private fun performRequest(analysis: VisualProductAnalysis, frame: CapturedCameraFrame): ProductSearchResult {
        val connection = (URL("$baseUrl/api/v1/products/search").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "application/json")
        }

        try {
            val body = JSONObject().apply {
                // Send the backend's own validated analysis back unchanged.
                put("analysis", JSONObject(analysis.rawJson))
                put("maxResults", MAX_RESULTS)
                put("imageBase64", Base64.encodeToString(frame.jpegBytes, Base64.NO_WRAP))
                put("mimeType", "image/jpeg")
                put("rotationDegrees", frame.rotationDegrees)
            }.toString()
            connection.outputStream.bufferedWriter(Charsets.UTF_8).use { it.write(body) }

            val statusCode = connection.responseCode
            val responseText = (if (statusCode in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader(Charsets.UTF_8)
                ?.use { it.readText().take(MAX_RESPONSE_CHARS) }
                .orEmpty()
            if (statusCode !in 200..299) {
                val detail = runCatching { JSONObject(responseText).optString("detail") }.getOrNull()
                throw ProductSearchException(
                    detail?.takeIf { it.isNotBlank() }
                        ?: "Product search returned HTTP $statusCode. Please retry.",
                )
            }
            return parseResult(JSONObject(responseText))
        } catch (exception: SocketTimeoutException) {
            throw ProductSearchException("Product search timed out. Check the connection and retry.", exception)
        } catch (exception: ProductSearchException) {
            throw exception
        } catch (exception: IOException) {
            throw ProductSearchException(
                "Cannot reach the local backend. Start it, check adb reverse, and retry.",
                exception,
            )
        } catch (exception: Exception) {
            throw ProductSearchException("The product search response was invalid. Please retry.", exception)
        } finally {
            connection.disconnect()
        }
    }

    private fun performDimensionRequest(product: ProductCandidate): ResolvedDimensions {
        val body = JSONObject().apply {
            put("productId", product.id)
            put("productUrl", product.productUrl)
        }
        val json = postJson("/api/v1/products/dimensions", body, DIMENSION_READ_TIMEOUT_MS, "Dimension lookup")
        val status = json.getString("status")
        require(status in setOf("verified", "partial", "unavailable"))
        val mappingJson = json.optJSONObject("axisMapping")
        return ResolvedDimensions(
            productId = json.getString("productId"),
            widthMeters = json.optionalDouble("widthMeters")?.takeIf { it > 0.0 },
            depthMeters = json.optionalDouble("depthMeters")?.takeIf { it > 0.0 },
            heightMeters = json.optionalDouble("heightMeters")?.takeIf { it > 0.0 },
            dimensionsMeters = json.optJSONArray("dimensionsMeters")?.let { values ->
                buildList {
                    for (index in 0 until values.length()) {
                        values.optDouble(index).takeIf { !it.isNaN() && it > 0.0 }?.let(::add)
                    }
                }
            }.orEmpty(),
            axisMapping = mappingJson?.let {
                DimensionAxisMapping(
                    widthIndex = it.optionalInt("widthIndex"),
                    depthIndex = it.optionalInt("depthIndex"),
                    heightIndex = it.optionalInt("heightIndex"),
                    confidence = it.optDouble("confidence", 0.0).takeIf { value -> !value.isNaN() } ?: 0.0,
                    reason = it.optionalString("reason"),
                    source = it.optionalString("source") ?: "none",
                )
            },
            status = status,
            sourceType = json.optionalString("sourceType") ?: "unavailable",
            sourceUrl = json.optionalString("sourceUrl"),
            sourceName = json.optionalString("sourceName"),
            rawDimensions = json.optionalString("rawDimensions"),
            sourcePath = json.optionalString("sourcePath"),
            extractionMethod = json.optionalString("extractionMethod"),
            variantScope = json.optionalString("variantScope"),
            retryable = json.optBoolean("retryable", false),
            message = json.optionalString("message"),
        )
    }

    private fun postJson(path: String, payload: JSONObject, readTimeoutMs: Int, label: String): JSONObject {
        val connection = (URL("$baseUrl$path").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = readTimeoutMs
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "application/json")
        }
        try {
            connection.outputStream.bufferedWriter(Charsets.UTF_8).use { it.write(payload.toString()) }
            val statusCode = connection.responseCode
            val responseText = (if (statusCode in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader(Charsets.UTF_8)
                ?.use { it.readText().take(MAX_RESPONSE_CHARS) }
                .orEmpty()
            if (statusCode !in 200..299) {
                val detail = runCatching { JSONObject(responseText).optString("detail") }.getOrNull()
                throw ProductSearchException(
                    detail?.takeIf { it.isNotBlank() } ?: "$label returned HTTP $statusCode. Please retry.",
                )
            }
            return JSONObject(responseText)
        } catch (exception: SocketTimeoutException) {
            throw ProductSearchException("$label timed out. Check the connection and retry.", exception)
        } catch (exception: ProductSearchException) {
            throw exception
        } catch (exception: IOException) {
            throw ProductSearchException(
                "Cannot reach the local backend. Start it, check adb reverse, and retry.",
                exception,
            )
        } catch (exception: Exception) {
            throw ProductSearchException("The $label response was invalid. Please retry.", exception)
        } finally {
            connection.disconnect()
        }
    }

    private fun parseResult(json: JSONObject): ProductSearchResult {
        val resultSource = json.getString("resultSource")
        require(resultSource == "live" || resultSource == "cache")
        val productsJson = json.getJSONArray("products")
        val products = buildList {
            for (index in 0 until productsJson.length()) {
                // Skip a single malformed entry rather than failing the whole list.
                runCatching { parseProduct(productsJson.getJSONObject(index)) }.getOrNull()?.let(::add)
            }
        }
        return ProductSearchResult(
            query = json.getString("query"),
            provider = json.getString("provider"),
            resultSource = resultSource,
            cachedAt = json.optionalString("cachedAt"),
            products = products,
            message = json.optionalString("message"),
            queries = json.optJSONArray("queries")?.let { array ->
                buildList {
                    for (index in 0 until array.length()) {
                        val item = array.optJSONObject(index) ?: continue
                        val query = item.optionalString("query") ?: continue
                        add(QueryOutcome(query, item.optionalString("status") ?: "ok", item.optInt("resultCount", 0)))
                    }
                }
            }.orEmpty(),
            retrievalMode = json.optionalString("retrievalMode") ?: "text_only",
            visualSearchStatus = json.optionalString("visualSearchStatus") ?: "skipped",
            timings = json.optJSONObject("timings")?.let { timing ->
                RetrievalTimings(
                    lensUploadMs = timing.optionalNonNegativeInt("lensUploadMs"),
                    lensSearchMs = timing.optionalNonNegativeInt("lensSearchMs"),
                    textSearchMs = timing.optionalNonNegativeInt("textSearchMs"),
                    mergeRerankMs = timing.optionalNonNegativeInt("mergeRerankMs"),
                    localVisualRerankMs = timing.optionalNonNegativeInt("localVisualRerankMs"),
                    totalMs = timing.optionalNonNegativeInt("totalMs"),
                )
            } ?: RetrievalTimings(),
        )
    }

    private fun parseProduct(json: JSONObject): ProductCandidate {
        val price = json.optionalDouble("price")?.takeIf { it >= 0.0 }
        val dimensions = json.optJSONObject("dimensions")
        return ProductCandidate(
            id = json.getString("id"),
            provider = json.getString("provider"),
            providerProductId = json.getString("providerProductId"),
            title = requireNotNull(json.optionalString("title")),
            price = price,
            priceText = json.optionalString("priceText"),
            currency = json.optionalString("currency"),
            retailer = json.optionalString("retailer"),
            imageUrl = json.optionalString("imageUrl")?.takeIf { it.startsWith("https://") || it.startsWith("http://") },
            productUrl = json.getString("productUrl"),
            rating = json.optionalDouble("rating"),
            reviewCount = json.optionalDouble("reviewCount")?.toInt(),
            inStock = if (json.has("inStock") && !json.isNull("inStock")) json.optBoolean("inStock") else null,
            dimensions = ProductDimensions(
                widthMeters = dimensions?.optionalDouble("widthMeters"),
                depthMeters = dimensions?.optionalDouble("depthMeters"),
                heightMeters = dimensions?.optionalDouble("heightMeters"),
                status = dimensions?.optionalString("status") ?: "unavailable",
                source = dimensions?.optionalString("source"),
            ),
            retrievalSources = json.optJSONArray("retrievalSources").toStringList(),
            textRank = json.optionalPositiveInt("textRank"),
            visualRank = json.optionalPositiveInt("visualRank"),
            visualSimilarityScore = json.optionalDouble("visualSimilarityScore"),
            combinedScore = json.optionalDouble("combinedScore"),
            identifiers = json.optJSONObject("identifiers")?.let { identifiers ->
                buildMap {
                    identifiers.keys().forEach { key ->
                        identifiers.optString(key).trim().takeIf { it.isNotEmpty() }?.let { put(key, it) }
                    }
                }
            }.orEmpty(),
        )
    }

    private fun JSONObject.optionalString(key: String): String? {
        if (!has(key) || isNull(key)) return null
        return optString(key).trim().takeIf { it.isNotEmpty() }
    }

    private fun JSONObject.optionalDouble(key: String): Double? {
        if (!has(key) || isNull(key)) return null
        return optDouble(key).takeIf { !it.isNaN() }
    }

    private fun JSONObject.optionalInt(key: String): Int? {
        if (!has(key) || isNull(key)) return null
        return optInt(key).takeIf { it in 0..2 }
    }

    private fun JSONObject.optionalPositiveInt(key: String): Int? {
        if (!has(key) || isNull(key)) return null
        return optInt(key).takeIf { it >= 1 }
    }

    private fun JSONObject.optionalNonNegativeInt(key: String): Int? {
        if (!has(key) || isNull(key)) return null
        return optInt(key).takeIf { it >= 0 }
    }

    private fun JSONArray?.toStringList(): List<String> {
        if (this == null) return emptyList()
        return buildList {
            for (index in 0 until length()) {
                optString(index).trim().takeIf { it.isNotEmpty() }?.let(::add)
            }
        }
    }

    private class ProductSearchException(message: String, cause: Throwable? = null) :
        IOException(message, cause)

    companion object {
        private const val MAX_RESULTS = 5
        private const val CONNECT_TIMEOUT_MS = 5_000
        // Lens modes run concurrently, followed by one bounded local visual rerank.
        private const val READ_TIMEOUT_MS = 240_000
        // Browser Use is a last-resort path and may legitimately run after the
        // faster structured/crawl/search stages. Do not abandon a valid result
        // while the backend's bounded fallback is still working.
        private const val DIMENSION_READ_TIMEOUT_MS = 360_000
        private const val MAX_RESPONSE_CHARS = 200_000
    }
}
