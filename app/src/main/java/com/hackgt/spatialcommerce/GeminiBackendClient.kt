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

class GeminiBackendClient(
    private val baseUrl: String = "http://127.0.0.1:8000",
) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()

    fun analyze(
        frame: CapturedCameraFrame,
        userRequest: String? = null,
        callback: (Result<VisualProductAnalysis>) -> Unit,
    ) {
        executor.execute {
            callback(runCatching { performRequest(frame, userRequest) })
        }
    }

    fun close() {
        executor.shutdownNow()
    }

    private fun performRequest(
        frame: CapturedCameraFrame,
        userRequest: String?,
    ): VisualProductAnalysis {
        val connection = (URL("$baseUrl/api/v1/analyze").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "application/json")
        }

        try {
            val body = JSONObject().apply {
                put("imageBase64", Base64.encodeToString(frame.jpegBytes, Base64.NO_WRAP))
                put("mimeType", "image/jpeg")
                put("rotationDegrees", frame.rotationDegrees)
                if (!userRequest.isNullOrBlank()) put("userRequest", userRequest.trim())
            }.toString()
            connection.outputStream.bufferedWriter(Charsets.UTF_8).use { it.write(body) }

            val statusCode = connection.responseCode
            val responseText = (if (statusCode in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader(Charsets.UTF_8)
                ?.use { it.readText().take(MAX_RESPONSE_CHARS) }
                .orEmpty()
            if (statusCode !in 200..299) {
                val detail = runCatching { JSONObject(responseText).optString("detail") }.getOrNull()
                throw BackendException(
                    detail?.takeIf { it.isNotBlank() }
                        ?: "Analysis service returned HTTP $statusCode.",
                )
            }
            return parseAnalysis(JSONObject(responseText))
        } catch (exception: SocketTimeoutException) {
            throw BackendException("Analysis timed out. Check the connection and retry.", exception)
        } catch (exception: BackendException) {
            throw exception
        } catch (exception: IOException) {
            throw BackendException(
                "Cannot reach the local analysis service. Start the backend and check adb reverse.",
                exception,
            )
        } catch (exception: Exception) {
            throw BackendException("The analysis response was invalid. Please retry.", exception)
        } finally {
            connection.disconnect()
        }
    }

    private fun parseAnalysis(json: JSONObject): VisualProductAnalysis {
        require(json.has("objectDetected"))
        val confidence = json.getDouble("confidence")
        require(confidence in 0.0..1.0)
        return VisualProductAnalysis(
            objectDetected = json.getBoolean("objectDetected"),
            category = json.optionalString("category"),
            subcategory = json.optionalString("subcategory"),
            color = json.optionalString("color"),
            materials = json.optJSONArray("materials").toStringList(),
            style = json.optJSONArray("style").toStringList(),
            shape = json.optionalString("shape"),
            searchKeywords = json.optJSONArray("searchKeywords").toStringList(),
            confidence = confidence,
            message = json.optionalString("message"),
        )
    }

    private fun JSONObject.optionalString(key: String): String? {
        if (!has(key) || isNull(key)) return null
        return optString(key).trim().takeIf { it.isNotEmpty() }
    }

    private fun JSONArray?.toStringList(): List<String> {
        if (this == null) return emptyList()
        return buildList {
            for (index in 0 until length()) {
                optString(index).trim().takeIf { it.isNotEmpty() }?.let(::add)
            }
        }
    }

    private class BackendException(message: String, cause: Throwable? = null) :
        IOException(message, cause)

    companion object {
        private const val CONNECT_TIMEOUT_MS = 5_000
        private const val READ_TIMEOUT_MS = 50_000
        private const val MAX_RESPONSE_CHARS = 100_000
    }
}
