package br.arena.gptmachado

import android.app.Activity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import java.io.File

class MainActivity : Activity() {

    companion object {
        init { System.loadLibrary("gptmachado") }
    }

    /** Carrega o GGUF (uma vez). Retorna true em caso de sucesso. */
    private external fun initModel(path: String): Boolean

    /** Continua o texto a partir do prompt. */
    private external fun complete(prompt: String, nPredict: Int, temp: Float, topK: Int, seed: Int): String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val prompt = findViewById<EditText>(R.id.prompt)
        val go = findViewById<Button>(R.id.go)
        val out = findViewById<TextView>(R.id.out)

        // Copia o modelo dos assets para o storage interno na primeira execução
        val modelFile = File(filesDir, "gpt-machado-q8_0.gguf")
        Thread {
            try {
                if (!modelFile.exists() || modelFile.length() == 0L) {
                    assets.open("gpt-machado-q8_0.gguf").use { input ->
                        modelFile.outputStream().use { output -> input.copyTo(output) }
                    }
                }
                val ok = initModel(modelFile.absolutePath)
                runOnUiThread {
                    Toast.makeText(this,
                        if (ok) "Modelo carregado (offline)" else "Falha ao carregar o modelo",
                        Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread { out.text = "Erro: ${e.message}" }
            }
        }.start()

        go.setOnClickListener {
            val p = prompt.text.toString().trim()
            if (p.isEmpty()) return@setOnClickListener
            go.isEnabled = false
            Thread {
                val texto = complete(p, nPredict = 200, temp = 0.85f, topK = 60, seed = (0..99999).random())
                runOnUiThread {
                    out.text = p + texto
                    go.isEnabled = true
                }
            }.start()
        }
    }
}
