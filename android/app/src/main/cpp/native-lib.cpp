// native-lib.cpp — ponte JNI mínima entre o app Android e o llama.cpp.
// O modelo (gpt-machado-q8_0.gguf) é carregado do storage interno e a
// inferência roda 100% offline na CPU do celular.

#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "llama.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "gptmachado", __VA_ARGS__)

static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_br_arena_gptmachado_MainActivity_initModel(JNIEnv *env, jobject, jstring jpath) {
    if (g_model != nullptr) return JNI_TRUE;

    const char *path = env->GetStringUTFChars(jpath, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;  // CPU-only: funciona em qualquer aparelho
    g_model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (g_model == nullptr) {
        LOGI("falha ao carregar modelo");
        return JNI_FALSE;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;                       // = contexto de treino
    cp.n_threads = 2;
    cp.n_threads_batch = 2;
    g_ctx = llama_init_from_model(g_model, cp);
    if (g_ctx == nullptr) return JNI_FALSE;

    LOGI("modelo carregado");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_br_arena_gptmachado_MainActivity_complete(JNIEnv *env, jobject,
                                               jstring jprompt, jint nPredict,
                                               jfloat temp, jint topK, jint seed) {
    if (g_model == nullptr || g_ctx == nullptr) {
        return env->NewStringUTF(" [modelo não carregado]");
    }

    const char *prompt = env->GetStringUTFChars(jprompt, nullptr);

    // Tokeniza o prompt (sem BOS: o modelo não usa token inicial especial)
    const int n_prompt_max = 128;
    std::vector<llama_token> tokens(n_prompt_max);
    const char *it = prompt;
    int n_tokens = 0;
    while (*it != '\0' && n_tokens < n_prompt_max) {
        llama_token t[8];
        int n = llama_tokenize(g_model, it, 1, t, 8, false, false);
        if (n <= 0) break;
        for (int i = 0; i < n && n_tokens < n_prompt_max; i++) tokens[n_tokens++] = t[i];
        it += 1;  // avança byte a byte; o BPE byte-level lida com UTF-8 por partes
    }
    env->ReleaseStringUTFChars(jprompt, prompt);
    if (n_tokens == 0) return env->NewStringUTF("");

    // Garante espaço para a geração dentro do contexto
    const int n_ctx = (int) llama_n_ctx(g_ctx);
    if (n_tokens + nPredict > n_ctx) {
        const int keep = n_ctx - nPredict - 1;
        tokens.erase(tokens.begin(), tokens.end() - keep);
        n_tokens = keep;
    }

    llama_sampler *smp = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smp, llama_sampler_init_top_k(topK));
    llama_sampler_chain_add(smp, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(smp, llama_sampler_init_dist(seed));

    std::string out;
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    for (int i = 0; i < nPredict; i++) {
        if (llama_decode(g_ctx, batch) != 0) break;
        llama_token t = llama_sampler_sample(smp, g_ctx, -1);
        if (t == 0) break;  // <|endoftext|>
        char buf[128];
        int n = llama_token_to_piece(g_model, t, buf, sizeof(buf), 0, false);
        if (n > 0) out.append(buf, (size_t) n);
        batch = llama_batch_get_one(&t, 1);
    }
    llama_sampler_free(smp);
    return env->NewStringUTF(out.c_str());
}
