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
    val price: Double,
    val priceText: String?,
    val currency: String?,
    val retailer: String?,
    val imageUrl: String?,
    val productUrl: String,
    val rating: Double?,
    val reviewCount: Int?,
    val dimensions: ProductDimensions,
) {
    fun displayPrice(): String = priceText
        ?: String.format(Locale.US, "%.2f%s", price, currency?.let { " $it" }.orEmpty())

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
) {
    val isCached: Boolean get() = resultSource == "cache"
}

/** Dimensions of the selected product, resolved by the backend from explicit source data only. */
data class ResolvedDimensions(
    val productId: String,
    val widthMeters: Double?,
    val depthMeters: Double?,
    val heightMeters: Double?,
    /** "verified", "partial", or "unavailable". */
    val status: String,
    /** "json_ld", "structured_metadata", "spec_table", "page_text", or "unavailable". */
    val sourceType: String,
    val sourceUrl: String?,
    val sourceName: String?,
    val rawDimensions: String?,
    val retryable: Boolean,
    val message: String?,
)
