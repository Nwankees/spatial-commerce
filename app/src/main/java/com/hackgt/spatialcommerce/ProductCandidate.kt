package com.hackgt.spatialcommerce

import java.util.Locale

/** Physical dimensions reported by the provider; never estimated on device. */
data class ProductDimensions(
    val widthMeters: Double?,
    val depthMeters: Double?,
    val heightMeters: Double?,
    /** "complete", "partial", or "unavailable", as reported by the backend. */
    val status: String,
    val source: String?,
)

/** A real purchasable product returned by the backend's product-search provider. */
data class ProductCandidate(
    val id: String,
    val provider: String,
    val providerProductId: String,
    val title: String,
    val price: Double?,
    val priceText: String?,
    val currency: String?,
    val retailer: String?,
    val imageUrl: String?,
    val productUrl: String,
    val rating: Double?,
    val reviewCount: Int?,
    val inStock: Boolean? = null,
    val dimensions: ProductDimensions,
    val retrievalSources: List<String> = emptyList(),
    val textRank: Int? = null,
    val visualRank: Int? = null,
    val visualSimilarityScore: Double? = null,
    val combinedScore: Double? = null,
    val identifiers: Map<String, String> = emptyMap(),
) {
    fun displayPrice(): String = priceText
        ?: price?.let { String.format(Locale.US, "%.2f%s", it, currency?.let { code -> " $code" }.orEmpty()) }
        ?: "Price unavailable"

    fun displayRetailerAndRating(): String {
        val parts = mutableListOf(retailer ?: "Retailer unavailable")
        rating?.let { value ->
            val reviews = reviewCount?.let { count -> " (${String.format(Locale.US, "%,d", count)})" }.orEmpty()
            parts += String.format(Locale.US, "★ %.1f%s", value, reviews)
        }
        return parts.joinToString("  •  ")
    }
}

data class ProductSearchResult(
    val query: String,
    val provider: String,
    /** "live" or "cache". */
    val resultSource: String,
    val cachedAt: String?,
    val products: List<ProductCandidate>,
    val message: String?,
    /** Every search the backend ran for this request (Milestone 5.5 multi-query retrieval). */
    val queries: List<QueryOutcome> = emptyList(),
    val retrievalMode: String = "text_only",
    val visualSearchStatus: String = "skipped",
    val timings: RetrievalTimings = RetrievalTimings(),
) {
    val isCached: Boolean get() = resultSource == "cache"
}

data class RetrievalTimings(
    val lensUploadMs: Int? = null,
    val lensSearchMs: Int? = null,
    val textSearchMs: Int? = null,
    val mergeRerankMs: Int? = null,
    val localVisualRerankMs: Int? = null,
    val totalMs: Int? = null,
)

data class QueryOutcome(
    val query: String,
    /** "ok", "empty", or "failed". */
    val status: String,
    val resultCount: Int,
)

/** Dimensions of the selected product, resolved by the backend from explicit source data only. */
data class ResolvedDimensions(
    val productId: String,
    val widthMeters: Double?,
    val depthMeters: Double?,
    val heightMeters: Double?,
    /** All retailer-verified values in source order, even when W/D/H order is unresolved. */
    val dimensionsMeters: List<Double>,
    val axisMapping: DimensionAxisMapping?,
    /** "verified", "partial", or "unavailable". */
    val status: String,
    /** "json_ld", "structured_metadata", "spec_table", "page_text", or "unavailable". */
    val sourceType: String,
    val sourceUrl: String?,
    val sourceName: String?,
    val rawDimensions: String?,
    val sourcePath: String?,
    val extractionMethod: String?,
    val variantScope: String?,
    val retryable: Boolean,
    val message: String?,
)

data class DimensionAxisMapping(
    val widthIndex: Int?,
    val depthIndex: Int?,
    val heightIndex: Int?,
    val confidence: Double,
    val reason: String?,
    /** structured_fields | labels | none */
    val source: String,
)
