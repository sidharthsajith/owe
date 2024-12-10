import os
from groq import Groq

from dotenv import load_dotenv

load_dotenv()


class GroqLLM:

    temperature = 0
    max_tokens = 1000
    top_p = 1
    model = "llama-3.1-70b-versatile"  # Default model

    def __init__(self, model: str):
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key is None:
            raise Exception(
                "Please set GROQ_API_KEY environment variable."
                "You can obtain API key from https://console.groq.com/keys"
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    @property
    def _default_params(self):
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "model": self.model,
        }

    def chat(self, prompt):
        params = {
            **self._default_params,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                }
            ],
        }
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
