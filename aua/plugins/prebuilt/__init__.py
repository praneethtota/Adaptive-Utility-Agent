"""
aua/plugins/prebuilt — Prebuilt ModelBackendPlugin implementations.

All plugins implement the ModelBackendPlugin protocol from aua.plugins.interfaces.
Load any of these via aua_config.yaml:

    plugins:
      model_backend:
        import_path: aua.plugins.prebuilt.openai_backend:OpenAIBackend
        config:
          api_key_secret: OPENAI_API_KEY
          model: gpt-4o

Available plugins:

    OpenAI-native:
        openai_backend.OpenAIBackend       — GPT-4o, GPT-4o mini (OpenAI SDK)
        anthropic_backend.AnthropicBackend — Claude Sonnet, Haiku (Anthropic SDK)
        google_backend.GoogleBackend       — Gemini 1.5 Pro, 2.0 Flash (Google REST API)

    OpenAI-compatible (same interface, different base URL):
        xai_backend.XAIBackend             — Grok-2 (api.x.ai/v1)
        mistral_backend.MistralBackend     — Mistral Large (api.mistral.ai/v1)
        groq_backend.GroqBackend           — Llama 3.3 70B (api.groq.com/openai/v1)
        deepseek_backend.DeepSeekBackend   — DeepSeek-V3, DeepSeek-R1 (api.deepseek.com/v1)

All plugins are contributed from AUA-Veritas (praneethtota/AUA-Veritas).
See tutorial §13 for full usage examples.
"""
