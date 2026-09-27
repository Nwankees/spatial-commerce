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
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit

data class ConversationContext(
    val shopperId: String,
    val analysis: VisualProductAnalysis?,
    val products: List<ProductCandidate>,
    val selectedProductId: String?,
    val availableWidthMeters: Double?,
    val availableDepthMeters: Double?,
    val frame: CapturedCameraFrame?,
)

data class ConversationProgress(val phase: String, val action: String?) {
    fun label(): String = when {
        phase == "planning" -> "Understanding your request with local Qwen…"
        phase == "responding" -> "Writing the answer…"
        action == "find_similar_products" || action == "refine_search" -> "Searching products…"
        action == "check_fit" -> "Checking fit…"
        action == "request_ar_preview" -> "Preparing the 3D preview…"
        action == "compare_products" -> "Comparing products…"
        action == "prepare_purchase" -> "Preparing a safe purchase review…"
        action == "confirm_purchase" -> "Verifying trusted-agent checkout…"
        else -> "Working on your request…"
    }
}

data class PurchaseReview(
    val id: String,
    val title: String,
    val merchant: String,
    val price: Double,
    val currency: String,
    val quantity: Int,
    val total: Double,
    val variant: String?,
    val fitStatus: String?,
    val status: String,
    val trustedAgentStatus: String,
    val isSimulation: Boolean,
    val checkoutUrl: String,
)

data class ConversationReply(
    val message: String,
    val status: String,
    val action: String,
    val planner: String,
    val products: List<ProductCandidate>,
    val searchResult: ProductSearchResult?,
    val selectedProduct: ProductCandidate?,
    val selectedProductId: String?,
    val uiDirective: String,
    val purchase: PurchaseReview?,
    val checkoutUrl: String?,
)

/** USB-local M7 API client. Secrets and the local model remain on the backend. */
class ConversationClient(
    private val baseUrl: String = "http://127.0.0.1:8000",
) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val progressExecutor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private var sessionId: String? = null

    fun send(
        message: String,
        context: ConversationContext,
        onProgress: (ConversationProgress) -> Unit = {},
        callback: (Result<ConversationReply>) -> Unit,
    ) {
        executor.execute {
            callback(runCatching { sendBlocking(message, context, onProgress) })
        }
    }

    fun reset() {
        executor.execute {
            val current = sessionId ?: return@execute
            runCatching {
                request("POST", "/api/v1/conversations/$current/reset", JSONObject(), SHORT_READ_TIMEOUT_MS)
            }
        }
    }

    fun close() {
        executor.shutdownNow()
        progressExecutor.shutdownNow()
    }

    private fun sendBlocking(
        message: String,
        context: ConversationContext,
        onProgress: (ConversationProgress) -> Unit,
    ): ConversationReply {
        var id = ensureSession()
        try {
            syncContext(id, context)
            return requestTurn(id, message, onProgress)
        } catch (exception: ConversationHttpException) {
            if (exception.statusCode != 404) throw exception
            // Backend restart only loses process-local M7 state. Recreate and replay the
            // current Android context once instead of making the user restart the app.
            sessionId = null
            id = ensureSession()
            syncContext(id, context)
            return requestTurn(id, message, onProgress)
        }
    }

    private fun requestTurn(
        id: String,
        message: String,
        onProgress: (ConversationProgress) -> Unit,
    ): ConversationReply {
        val polling = progressExecutor.scheduleAtFixedRate(
            { pollProgress(id, onProgress) },
            PROGRESS_INITIAL_DELAY_MS,
            PROGRESS_INTERVAL_MS,
            TimeUnit.MILLISECONDS,
        )
        return try {
            parseReply(request(
                "POST",
                "/api/v1/conversations/$id/messages",
                JSONObject().put("message", message),
                TURN_READ_TIMEOUT_MS,
            ))
        } finally {
            polling.cancel(true)
        }
    }

    private fun pollProgress(id: String, onProgress: (ConversationProgress) -> Unit) {
        val state = runCatching {
            request("GET", "/api/v1/conversations/$id", null, PROGRESS_READ_TIMEOUT_MS)
                .getJSONObject("state")
        }.getOrNull() ?: return
        val phase = state.optionalString("activePhase") ?: return
        onProgress(ConversationProgress(phase, state.optionalString("activeAction")))
    }

    private fun ensureSession(): String {
        sessionId?.let { return it }
        val created = request(
            "POST", "/api/v1/conversations", JSONObject(), SHORT_READ_TIMEOUT_MS,
        )
        return created.getString("sessionId").also { sessionId = it }
    }

    private fun syncContext(id: String, context: ConversationContext) {
        val body = JSONObject().apply {
            put("shopperId", context.shopperId)
            put("analysis", context.analysis?.let { JSONObject(it.rawJson) } ?: JSONObject.NULL)
            put(
                "imageBase64",
                context.frame?.let { Base64.encodeToString(it.jpegBytes, Base64.NO_WRAP) } ?: JSONObject.NULL,
            )
            put("mimeType", "image/jpeg")
            put("rotationDegrees", context.frame?.rotationDegrees ?: 0)
            put("products", JSONArray().apply { context.products.forEach { put(productJson(it)) } })
            put("selectedProductId", context.selectedProductId ?: JSONObject.NULL)
            put(
                "measuredSpace",
                if (context.availableWidthMeters != null && context.availableDepthMeters != null) {
                    JSONObject()
                        .put("widthMeters", context.availableWidthMeters)
                        .put("depthMeters", context.availableDepthMeters)
                } else {
                    JSONObject.NULL
                },
            )
        }
        request("PUT", "/api/v1/conversations/$id/context", body, CONTEXT_READ_TIMEOUT_MS)
    }

    private fun request(
        method: String,
        path: String,
        payload: JSONObject?,
        readTimeoutMs: Int,
    ): JSONObject {
        val connection = (URL("$baseUrl$path").openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = readTimeoutMs
            doOutput = payload != null
            if (payload != null) setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "application/json")
        }
        try {
            payload?.let { body ->
                connection.outputStream.bufferedWriter(Charsets.UTF_8).use { it.write(body.toString()) }
            }
            val statusCode = connection.responseCode
            val responseText = (if (statusCode in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader(Charsets.UTF_8)
                ?.use { it.readText().take(MAX_RESPONSE_CHARS) }
                .orEmpty()
            if (statusCode !in 200..299) {
                val detail = runCatching { JSONObject(responseText).optString("detail") }.getOrNull()
                throw ConversationHttpException(
                    statusCode,
                    detail?.takeIf { it.isNotBlank() } ?: "Shopping assistant returned HTTP $statusCode.",
                )
            }
            return JSONObject(responseText)
        } catch (exception: SocketTimeoutException) {
            throw IOException("The local shopping assistant timed out. Please retry.", exception)
        } catch (exception: ConversationHttpException) {
            throw exception
        } catch (exception: IOException) {
            throw IOException("Cannot reach the local assistant. Start the backend and check adb reverse.", exception)
        } finally {
            connection.disconnect()
        }
    }

    private fun parseReply(json: JSONObject): ConversationReply {
        val products = json.optJSONArray("products")?.let(::parseProducts).orEmpty()
        val searchResult = json.optJSONObject("searchResult")?.let(::parseSearchResult)
        val selected = json.optJSONObject("selectedProduct")?.let(::parseProduct)
        val state = json.getJSONObject("state")
        return ConversationReply(
            message = json.getString("message"),
            status = json.getString("status"),
            action = json.getJSONObject("action").getString("action"),
            planner = json.getString("planner"),
            products = products,
            searchResult = searchResult,
            selectedProduct = selected,
            selectedProductId = state.optionalString("selectedProductId"),
            uiDirective = json.optString("uiDirective", "none"),
            purchase = (json.optJSONObject("purchase") ?: state.optJSONObject("purchase"))?.let(::parsePurchase),
            checkoutUrl = json.optionalString("checkoutUrl"),
        )
    }

    private fun parsePurchase(json: JSONObject): PurchaseReview {
        val variants = json.optJSONObject("selectedVariant")
        val variantText = variants?.let { value ->
            buildList {
                value.keys().forEach { key ->
                    value.optString(key).trim().takeIf { it.isNotEmpty() }?.let { add("$key: $it") }
                }
            }.joinToString(", ").ifBlank { null }
        }
        return PurchaseReview(
            id = json.getString("id"),
            title = json.getString("title"),
            merchant = json.getString("merchant"),
            price = json.getDouble("price"),
            currency = json.getString("currency"),
            quantity = json.getInt("quantity"),
            total = json.getDouble("total"),
            variant = variantText,
            fitStatus = json.optionalString("fitStatus"),
            status = json.getString("status"),
            trustedAgentStatus = json.getString("trustedAgentStatus"),
            isSimulation = json.optBoolean("isSimulation", true),
            checkoutUrl = json.getString("checkoutUrl"),
        )
    }

    private fun parseProducts(array: JSONArray): List<ProductCandidate> = buildList {
        for (index in 0 until array.length()) {
            array.optJSONObject(index)?.let { json -> runCatching { parseProduct(json) }.getOrNull()?.let(::add) }
        }
    }

    private fun parseSearchResult(json: JSONObject): ProductSearchResult {
        val resultSource = json.getString("resultSource")
        require(resultSource == "live" || resultSource == "cache")
        return ProductSearchResult(
            query = json.getString("query"),
            provider = json.getString("provider"),
            resultSource = resultSource,
            cachedAt = json.optionalString("cachedAt"),
            products = json.optJSONArray("products")?.let(::parseProducts).orEmpty(),
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
        val dimensions = json.optJSONObject("dimensions")
        return ProductCandidate(
            id = json.getString("id"),
            provider = json.getString("provider"),
            providerProductId = json.getString("providerProductId"),
            title = json.getString("title"),
            price = json.optionalDouble("price")?.takeIf { it >= 0.0 },
            priceText = json.optionalString("priceText"),
            currency = json.optionalString("currency"),
            retailer = json.optionalString("retailer"),
            imageUrl = json.optionalString("imageUrl"),
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

    private fun productJson(product: ProductCandidate) = JSONObject().apply {
        put("id", product.id)
        put("provider", product.provider)
        put("providerProductId", product.providerProductId)
        put("title", product.title)
        put("price", product.price ?: JSONObject.NULL)
        put("priceText", product.priceText ?: JSONObject.NULL)
        put("currency", product.currency ?: JSONObject.NULL)
        put("retailer", product.retailer ?: JSONObject.NULL)
        put("imageUrl", product.imageUrl ?: JSONObject.NULL)
        put("productUrl", product.productUrl)
        put("rating", product.rating ?: JSONObject.NULL)
        put("reviewCount", product.reviewCount ?: JSONObject.NULL)
        put("dimensions", JSONObject().apply {
            put("widthMeters", product.dimensions.widthMeters ?: JSONObject.NULL)
            put("depthMeters", product.dimensions.depthMeters ?: JSONObject.NULL)
            put("heightMeters", product.dimensions.heightMeters ?: JSONObject.NULL)
            put("status", product.dimensions.status)
            put("source", product.dimensions.source ?: JSONObject.NULL)
        })
    }

    private fun JSONObject.optionalString(key: String): String? =
        if (!has(key) || isNull(key)) null else optString(key).trim().takeIf { it.isNotEmpty() }

    private fun JSONObject.optionalDouble(key: String): Double? =
        if (!has(key) || isNull(key)) null else optDouble(key).takeIf { !it.isNaN() }

    private fun JSONObject.optionalPositiveInt(key: String): Int? =
        if (!has(key) || isNull(key)) null else optInt(key).takeIf { it >= 1 }

    private fun JSONObject.optionalNonNegativeInt(key: String): Int? =
        if (!has(key) || isNull(key)) null else optInt(key).takeIf { it >= 0 }

    private fun JSONArray?.toStringList(): List<String> {
        if (this == null) return emptyList()
        return buildList {
            for (index in 0 until length()) {
                optString(index).trim().takeIf { it.isNotEmpty() }?.let(::add)
            }
        }
    }

    private class ConversationHttpException(val statusCode: Int, message: String) : IOException(message)

    companion object {
        private const val CONNECT_TIMEOUT_MS = 5_000
        private const val SHORT_READ_TIMEOUT_MS = 15_000
        private const val CONTEXT_READ_TIMEOUT_MS = 30_000
        // Only a message turn gets the long envelope. It runs off the UI thread,
        // while typed progress is polled separately and the AR controls stay usable.
        private const val TURN_READ_TIMEOUT_MS = 480_000
        private const val PROGRESS_READ_TIMEOUT_MS = 8_000
        private const val PROGRESS_INITIAL_DELAY_MS = 500L
        private const val PROGRESS_INTERVAL_MS = 900L
        private const val MAX_RESPONSE_CHARS = 300_000
    }
}
