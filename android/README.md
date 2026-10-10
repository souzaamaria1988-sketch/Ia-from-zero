# 📱 App Android do GPT-Machado (offline, IA dentro do APK)

App mínimo (Kotlin + JNI + llama.cpp) que embute o GGUF de **1,58 MB** e roda
**100% offline** — sem servidor, sem internet, sem permissões.

```
usuário digita "— Capitu," → modelo roda na CPU do celular → continuação em estilo Machado
```

## Arquitetura

| Camada | Arquivo | Papel |
|---|---|---|
| UI | `app/src/main/java/br/arena/gptmachado/MainActivity.kt` | Tela + cópia do modelo dos assets + thread de geração |
| Ponte nativa | `app/src/main/cpp/native-lib.cpp` | JNI: carrega o GGUF, tokeniza, amostra (top-k + temp), detokeniza |
| Motor | `llama.cpp` (clonado pelo script) | Inferência quantizada Q8_0 na CPU |
| Modelo | `app/src/main/assets/gpt-machado-q8_0.gguf` | 1,58 MB, embutido no APK |

## Como compilar

Pré-requisitos: [Android Studio](https://developer.android.com/studio) (com NDK e CMake pelo SDK Manager).

```bash
# 1. baixa o código do llama.cpp (versão fixa v0.6.0)
bash android/scripts/fetch-llama-cpp.sh

# 2. embute o modelo no APK
bash android/scripts/colocar-modelo.sh

# 3. abra a pasta android/ no Android Studio → Run ▶️
```

Ou por linha de comando:

```bash
cd android
./gradlew assembleDebug          # APK em app/build/outputs/apk/debug/
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Notas técnicas

- `minSdk 24` (Android 7.0+) — cobre praticamente todos os aparelhos em uso.
- ABIs: `arm64-v8a` (celulares reais) e `x86_64` (emulador).
- Contexto de 256 tokens (igual ao treino), amostragem top-k 60 / temp 0.85.
- O binário nativo ocupa ~2–3 MB por ABI; o APK final fica em torno de **6 MB** —
  menor que a maioria das fotos que você manda por mensagem.
