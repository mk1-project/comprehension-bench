import os
from together import Together
from anthropic import Anthropic
from openai import OpenAI
from google import genai
from tenacity import retry, wait_exponential

class Model:
    """
    A unified interface for interacting with different LLM providers.
    Supports Together AI, Google, Anthropic, and OpenAI backends.
    """
    def __init__(self, backend, model, api_key=None):
        """
        Initialize the Model with the specified backend and model.

        Args:
            backend (str): The LLM provider ('together', 'google', 'anthropic', or 'openai').
            model (str): The specific model to use from the chosen provider.
            api_key (str, optional): API key for the provider. If not provided, will attempt
                                     to load from environment variables.
        """
        self.model = model
        self.api_key = api_key
        self.url = None

        if backend == "together":
            # Setup Together AI client
            if self.api_key is None:
                self.api_key = os.getenv("TOGETHER_API_KEY")
                if not self.api_key:
                    raise ValueError("TOGETHER_API_KEY is not set")
            self.client = Together(api_key=self.api_key)
            self.generate = self.generate_together

        elif backend == "google":
            # Setup Google Gemini client
            if self.api_key is None:
                self.api_key = os.getenv("GOOGLE_API_KEY")
                if not self.api_key:
                    raise ValueError("GOOGLE_API_KEY is not set")
            self.client = genai.Client(api_key=self.api_key)
            self.generate = self.generate_google

        elif backend == "anthropic":
            # Setup Anthropic client
            if self.api_key is None:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                if not self.api_key:
                    raise ValueError("ANTHROPIC_API_KEY is not set")
            self.client = Anthropic(api_key=self.api_key)
            self.generate = self.generate_anthropic

        elif backend == "openai":
            # Setup OpenAI client
            if self.api_key is None:
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY is not set")
            self.client = OpenAI(api_key=self.api_key)
            self.generate = self.generate_openai

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate_together(self, user_content, temperature=0.3, max_tokens=8192):
        """
        Generate text using Together AI models with automatic retries.

        Args:
            user_content (str): The prompt to send to the model
            temperature (float, optional): Controls randomness. Lower is more deterministic.
            max_tokens (int, optional): Maximum number of tokens to generate

        Returns:
            str: The generated text response
        """
        messages = [{"role": "user", "content": user_content}]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = self.client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content

        return response_text

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate_anthropic(self, user_content, temperature=0.3, max_tokens=8192):
        """
        Generate text using Anthropic Claude models with automatic retries.

        Args:
            user_content (str): The prompt to send to the model
            temperature (float, optional): Controls randomness. Lower is more deterministic.
            max_tokens (int, optional): Maximum number of tokens to generate

        Returns:
            str: The generated text response
        """
        messages = [{"role": "user", "content": user_content}]

        kwargs = {
            'model': self.model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': messages
        }
        response = self.client.messages.create(**kwargs)
        response_text = response.content[0].text

        return response_text

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate_openai(self, user_content, temperature=0.3, max_tokens=8192):
        """
        Generate text using OpenAI models with automatic retries.

        Args:
            user_content (str): The prompt to send to the model
            temperature (float, optional): Controls randomness. Lower is more deterministic.
            max_tokens (int, optional): Maximum number of tokens to generate

        Returns:
            str: The generated text response
        """
        messages = [{"role": "user", "content": user_content}]
        kwargs = {
            'model': self.model,
            'messages': messages,
            'max_completion_tokens': max_tokens,
        }

        # OpenAI reasoning models don't support temperature
        no_temperature_models = ['o3-mini', 'o1', 'o1-mini', 'o1-pro']
        if self.model not in no_temperature_models:
            kwargs['temperature'] = temperature

        response = self.client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content

        return response_text

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate_google(self, user_content, temperature=0.3, max_tokens=8192):
        """
        Generate text using Google Gemini models with automatic retries.

        Args:
            user_content (str): The prompt to send to the model
            temperature (float, optional): Controls randomness. Lower is more deterministic.
            max_tokens (int, optional): Maximum number of tokens to generate

        Returns:
            str: The generated text response
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )
        return response.text
