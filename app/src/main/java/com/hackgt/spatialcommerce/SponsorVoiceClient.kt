package com.hackgt.spatialcommerce

import android.media.MediaDataSource
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
import android.util.Base64
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Optional, fail-open ElevenLabs bridge. Text chat remains the source of truth; this class only
 * asks the backend to speak a reply after that reply is already visible.
 */
class SponsorVoiceClient(
    private val baseUrl: String = "http://127.0.0.1:8000",
) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val playback = VoicePlaybackController()
    private val generation = AtomicInteger(0)
    @Volatile private var closed = false

    fun speak(text: String, onUnavailable: (String) -> Unit = {}) {
        val cleaned = text.trim().take(MAX_TEXT_CHARS)
        if (cleaned.isEmpty() || closed) return
        val requestGeneration = generation.incrementAndGet()
        playback.stop()
        executor.execute {
            val result = runCatching { requestAudio(cleaned) }
            mainHandler.post {
                if (closed || requestGeneration != generation.get()) return@post
                result.fold(
                    onSuccess = { response ->
                        if (response.audio == null) {
                            onUnavailable(response.reason ?: "Voice is unavailable; the text reply is still ready.")
                        } else {
                            playback.play(response.audio) {
                                if (!closed && requestGeneration == generation.get()) {
                                    onUnavailable("Voice playback failed; the text reply is still ready.")
                                }
                            }
                        }
                    },
                    onFailure = {
                        onUnavailable("Voice is unavailable; the text reply is still ready.")
                    },
                )
            }
        }
    }

    fun stop() {
        generation.incrementAndGet()
        playback.stop()
    }

    fun close() {
        if (closed) return
        closed = true
        generation.incrementAndGet()
        executor.shutdownNow()
        playback.close()
    }

    private fun requestAudio(text: String): VoiceAudioResponse {
        val connection = (URL("$baseUrl/api/v1/sponsor/voice").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "application/json")
        }
        try {
            val request = JSONObject()
                .put("text", text)
                .put("enabled", true)
                .toString()
            connection.outputStream.bufferedWriter(Charsets.UTF_8).use { it.write(request) }

            val statusCode = connection.responseCode
            val responseText = (if (statusCode in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader(Charsets.UTF_8)
                ?.use { it.readText().take(MAX_RESPONSE_CHARS) }
                .orEmpty()
            if (statusCode !in 200..299) throw IOException("Voice service returned HTTP $statusCode.")

            val response = JSONObject(responseText)
            val available = response.optBoolean("voiceAvailable", false)
            val encoded = response.optString("audioBase64").takeIf { it.isNotBlank() }
            val audio = if (available && encoded != null) {
                Base64.decode(encoded, Base64.DEFAULT).takeIf { it.isNotEmpty() }
            } else {
                null
            }
            return VoiceAudioResponse(
                audio = audio,
                reason = response.optString("reason").trim().takeIf { it.isNotEmpty() },
            )
        } catch (exception: SocketTimeoutException) {
            throw IOException("Voice service timed out.", exception)
        } finally {
            connection.disconnect()
        }
    }

    private data class VoiceAudioResponse(val audio: ByteArray?, val reason: String?)

    companion object {
        private const val CONNECT_TIMEOUT_MS = 5_000
        private const val READ_TIMEOUT_MS = 30_000
        private const val MAX_TEXT_CHARS = 1_000
        private const val MAX_RESPONSE_CHARS = 8_000_000
    }
}

/** Main-thread, in-memory MP3 playback so synthesized speech never needs filesystem storage. */
private class VoicePlaybackController {
    private var player: MediaPlayer? = null
    private var dataSource: ByteArrayMediaDataSource? = null

    fun play(audio: ByteArray, onError: () -> Unit) {
        stop()
        val source = ByteArrayMediaDataSource(audio)
        val next = MediaPlayer()
        dataSource = source
        player = next
        try {
            next.setDataSource(source)
            next.setOnPreparedListener { prepared -> prepared.start() }
            next.setOnCompletionListener { completed -> release(completed) }
            next.setOnErrorListener { failed, _, _ ->
                release(failed)
                onError()
                true
            }
            next.prepareAsync()
        } catch (_: Exception) {
            release(next)
            onError()
        }
    }

    fun stop() {
        player?.let { current ->
            runCatching { current.stop() }
            release(current)
        }
    }

    fun close() = stop()

    private fun release(target: MediaPlayer) {
        if (player !== target) return
        target.reset()
        target.release()
        player = null
        dataSource?.close()
        dataSource = null
    }
}

private class ByteArrayMediaDataSource(private val bytes: ByteArray) : MediaDataSource() {
    override fun readAt(position: Long, buffer: ByteArray, offset: Int, size: Int): Int {
        if (position < 0 || position >= bytes.size) return -1
        val count = minOf(size, bytes.size - position.toInt())
        bytes.copyInto(buffer, offset, position.toInt(), position.toInt() + count)
        return count
    }

    override fun getSize(): Long = bytes.size.toLong()

    override fun close() = Unit
}
