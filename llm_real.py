"""Connect a real model. Works with any server that speaks the OpenAI chat-completions format:
  - Ollama (local):     base_url = http://localhost:11434/v1   (no key needed)
  - A hosted API:       base_url = the provider's /v1 address, key in an environment variable
No extra packages needed (standard library only)."""
import json
import re
import time
import urllib.error
import urllib.request


class OpenAICompatLLM:
    def __init__(self, base_url, model, api_key=None, temperature=0.0, timeout=900, max_tokens=500, json_mode=True):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key, self.temp, self.timeout, self.max_tokens = model, api_key, temperature, timeout, max_tokens
        self.json_mode = json_mode

    def complete(self, prompt, **kw):  # extra kwargs (scenario, mode) are ignored on purpose
        body = {"model": self.model, "temperature": self.temp, "max_tokens": kw.get("max_tokens", self.max_tokens),
                "messages": [{"role": "user", "content": prompt}]}
        if self.json_mode and kw.get("max_tokens") is None:
            body["response_format"] = {"type": "json_object"}  # only for the final JSON answers
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        last = None
        for attempt in range(2):
            try:
                req = urllib.request.Request(self.url, json.dumps(body).encode(), headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                text = data["choices"][0]["message"]["content"] or ""
                u = data.get("usage") or {}
                usage = {"in": u.get("prompt_tokens") or int(len(prompt.split()) * 1.3),
                         "out": u.get("completion_tokens") or int(len(text.split()) * 1.3)}
                return self._clean(text), usage
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
                time.sleep(2)
        raise RuntimeError(f"LLM call failed after retry: {last}")

    @staticmethod
    def _clean(text):
        """Models often wrap JSON in ```json fences or add a sentence around it. Keep the first {...} block."""
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return m.group(0) if m else text
