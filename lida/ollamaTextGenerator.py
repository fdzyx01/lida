from typing import Union, List, Dict
from llmx import TextGenerator
from llmx.datamodel import Message, TextGenerationConfig, TextGenerationResponse
from llmx.utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages
import os
import requests
from dataclasses import asdict
from lida.config import OLLAMA_API_URL, OLLAMA_API_KEY, OLLAMA_DEFAULT_MODEL, OLLAMA_MAX_TOKENS


class OllamaTextGenerator(TextGenerator):
    def __init__(
        self,
        api_url: str = OLLAMA_API_URL,
        api_key: str = OLLAMA_API_KEY,
        provider: str = "ollama",
        model: str = OLLAMA_DEFAULT_MODEL,
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", None)

        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        self.model_name = model or "default-model"
        self.model_max_token_dict = get_models_maxtoken_dict(models)

    def generate(
        self,
        messages: Union[List[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name
        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(
            self.model_max_token_dict.get(
                model, OLLAMA_MAX_TOKENS) - prompt_tokens - 10, 200
        )

        ollama_payload = {
            "model": model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "n": config.n,
        }

        cache_key_params = (ollama_payload) | {"messages": messages}
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=ollama_payload,
                headers=self.headers,
            )
            response.raise_for_status()
            api_response = response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to connect to Ollama API: {e}")

        # 处理 API 响应
        response = TextGenerationResponse(
            text=[Message(**choice["message"]) for choice in api_response["choices"]],
            logprobs=[],
            config=ollama_payload,
            usage=api_response.get("usage", {}),
        )
        # 缓存响应
        if use_cache:
            cache_request(
                cache=self.cache, params=cache_key_params, values=asdict(response)
            )
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
