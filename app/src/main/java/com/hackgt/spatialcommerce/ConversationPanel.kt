package com.hackgt.spatialcommerce

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** Small additive chat sheet; the original AR and button controls stay untouched underneath. */
class ConversationPanel(
    context: Context,
    private val onSend: (String) -> Unit,
    private val onClose: () -> Unit,
) : LinearLayout(context) {
    private val transcript: TextView
    private val scroll: ScrollView
    private val input: EditText
    private val sendButton: Button
    private val progress: TextView

    init {
        orientation = VERTICAL
        setBackgroundColor(Color.argb(248, 22, 24, 31))
        setPadding(dp(14), dp(10), dp(14), dp(14))
        visibility = View.GONE

        val title = TextView(context).apply {
            text = "Shopping assistant"
            setTextColor(Color.WHITE)
            textSize = 18f
            setTypeface(typeface, Typeface.BOLD)
        }
        val close = Button(context).apply {
            text = "Close"
            isAllCaps = false
            setOnClickListener { onClose() }
        }
        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(title, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
            addView(close, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
        })

        transcript = TextView(context).apply {
            text = "Assistant: Ask me to find similar products, refine results, check fit, or preview one in AR."
            setTextColor(Color.WHITE)
            textSize = 15f
            setPadding(0, dp(8), 0, dp(8))
        }
        scroll = ScrollView(context).apply {
            isFillViewport = true
            addView(transcript, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        }
        addView(scroll, LayoutParams(LayoutParams.MATCH_PARENT, 0, 1f))

        progress = TextView(context).apply {
            setTextColor(Color.rgb(190, 196, 208))
            textSize = 13f
            visibility = View.GONE
        }
        addView(progress)

        input = EditText(context).apply {
            hint = "e.g. Would the second one fit here?"
            setHintTextColor(Color.rgb(150, 155, 166))
            setTextColor(Color.WHITE)
            setSingleLine(true)
            setBackgroundColor(Color.rgb(39, 42, 52))
            setPadding(dp(12), 0, dp(12), 0)
        }
        sendButton = Button(context).apply {
            text = "Send"
            isAllCaps = false
            setOnClickListener { submit() }
        }
        addView(LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(input, LayoutParams(0, dp(52), 1f))
            addView(sendButton, LayoutParams(dp(88), dp(52)).apply { marginStart = dp(8) })
        })
    }

    fun open() {
        visibility = View.VISIBLE
        input.requestFocus()
        context.getSystemService(InputMethodManager::class.java)?.showSoftInput(input, 0)
    }

    fun close() {
        visibility = View.GONE
        context.getSystemService(InputMethodManager::class.java)?.hideSoftInputFromWindow(windowToken, 0)
    }

    fun showWorking(label: String = "Thinking with local Qwen…") {
        progress.text = label
        progress.visibility = View.VISIBLE
        sendButton.isEnabled = false
        input.isEnabled = false
    }

    fun showReply(message: String) {
        progress.visibility = View.GONE
        sendButton.isEnabled = true
        input.isEnabled = true
        append("Assistant", message)
        input.requestFocus()
    }

    fun showError(message: String) {
        progress.visibility = View.GONE
        sendButton.isEnabled = true
        input.isEnabled = true
        append("Assistant", message)
    }

    private fun submit() {
        val message = input.text.toString().trim()
        if (message.isEmpty() || !sendButton.isEnabled) return
        input.setText("")
        append("You", message)
        showWorking()
        onSend(message)
    }

    private fun append(speaker: String, message: String) {
        transcript.append("\n\n$speaker: $message")
        scroll.post { scroll.fullScroll(FOCUS_DOWN) }
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
}
