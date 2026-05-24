import os
import re
import torch
from typing import List
import shared.config as config


class LLMBackend:
    def __init__(self):
        self.use_gemini = config.USE_GEMINI
        self.use_openai = config.USE_OPENAI and not self.use_gemini
        self.device = self._auto_device()
        self.tokenizer = None
        self.model = None
        self.openai_client = None
        self.gemini_client = None
        self.sentence_model = None

        self.use_kimi = config.USE_KIMI and not self.use_gemini

        if self.use_gemini:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
                print(f"Loaded Gemini model {config.GEMINI_MODEL}")
            except Exception as e:
                print(f"Gemini init failed: {e}, falling back to local")
                self.use_gemini = False

        if self.use_kimi and not self.use_gemini:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(
                    api_key=config.KIMI_API_KEY,
                    base_url="https://api.moonshot.ai/v1",
                )
                print(f"Loaded Kimi model {config.KIMI_MODEL}")
            except Exception as e:
                print(f"Kimi init failed: {e}, falling back to local")
                self.use_kimi = False

        if not self.use_gemini and not self.use_kimi and self.use_openai:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            except Exception as e:
                print(f"OpenAI init failed: {e}, falling back to local")
                self.use_openai = False

        if not self.use_gemini and not self.use_kimi and not self.use_openai:
            self._load_local_model()

        self._load_embedding_model()

    def _auto_device(self):
        if config.DEVICE != "auto":
            return config.DEVICE
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_local_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                quantization_config=bnb_config,
                dtype=torch.float16,
                trust_remote_code=True,
            ).to(self.device)
            print(f"Loaded local model {config.MODEL_NAME} on {self.device}")
        except Exception as e:
            print(f"Failed to load local model: {e}")
            self.model = None

    def _load_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer(config.EMBEDDING_MODEL, device=self.device)
            print(f"Loaded embedding model {config.EMBEDDING_MODEL}")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            self.sentence_model = None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if self.use_gemini and self.gemini_client:
            return self._generate_gemini(prompt, max_tokens, temperature)
        if (self.use_kimi or self.use_openai) and self.openai_client:
            model = config.KIMI_MODEL if self.use_kimi else "gpt-3.5-turbo"
            return self._generate_openai(prompt, max_tokens, temperature, model=model)
        return self._generate_local(prompt, max_tokens, temperature)

    def _generate_gemini(self, prompt: str, max_tokens: int, temperature: float) -> str:
        import time
        import re
        from google.genai import types
        for attempt in range(4):
            try:
                response = self.gemini_client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                return response.text.strip()
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    m = re.search(r"retryDelay.*?(\d+)s", msg)
                    wait = int(m.group(1)) if m else 60
                    print(f"Gemini rate limit, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Gemini generation error: {e}")
                    return self._fallback_response(prompt)
        return self._fallback_response(prompt)

    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float, model: str = "gpt-3.5-turbo") -> str:
        import time
        for attempt in range(3):
            try:
                resp = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e)
                if "content_filter" in msg or "high risk" in msg:
                    # Prompt triggered content filter — not retryable
                    return self._fallback_response(prompt)
                if "429" in msg or "rate_limit" in msg.lower():
                    wait = 60 * (attempt + 1)
                    time.sleep(wait)
                    continue
                print(f"LLM generation error (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return self._fallback_response(prompt)
        return self._fallback_response(prompt)

    def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if self.model is None or self.tokenizer is None:
            return self._fallback_response(prompt)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            return text.strip()
        except Exception as e:
            print(f"Local generation error: {e}")
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        # Graceful fallback: try to extract what type of output is needed
        lower = prompt.lower()
        if "only a number" in lower or "output only" in lower:
            return "3.0"
        if "genre" in lower or "match" in lower:
            return "general interest"
        if "score" in lower and "0-10" in lower:
            return "5"
        return "I don't have enough information to provide a detailed response at this time."

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.sentence_model is None:
            return [[0.0] * 384 for _ in texts]
        try:
            embeddings = self.sentence_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            return [[0.0] * 384 for _ in texts]

    def is_ready(self) -> bool:
        if self.use_gemini:
            return self.gemini_client is not None
        if self.use_kimi or self.use_openai:
            return self.openai_client is not None
        return self.model is not None and self.tokenizer is not None


# Singleton instance
_llm_backend = None

def get_llm() -> LLMBackend:
    global _llm_backend
    if _llm_backend is None:
        _llm_backend = LLMBackend()
    return _llm_backend
