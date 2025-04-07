import re
import os
import json
import httpx
import asyncio
import anthropic
import openai
from openai import OpenAI
from google import genai
from google.genai import types
import requests
from typing import Dict, Any, Union, List
from tenacity import retry, wait_exponential

class Model:
    def __init__(self, backend, model, api_key=None):
        self.model = model
        self.api_key = api_key

        if backend == "google":
            if self.api_key is None:
                self.api_key = os.getenv("GOOGLE_API_KEY")
                if not self.api_key:
                    raise ValueError("GOOGLE_API_KEY is not set")
            self.client = genai.Client(api_key=self.api_key)
            self.generate = self.generate_google

        if backend == "anthropic":
            if self.api_key is None:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                if not self.api_key:
                    raise ValueError("ANTHROPIC_API_KEY is not set")
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.generate = self.generate_anthropic

        if backend == "openai":
            if self.api_key is None:
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY is not set")
            # Create a client with retry configuration
            self.client = OpenAI(
                api_key=self.api_key,
                max_retries=3,  # Number of retries
                timeout=60.0    # Timeout in seconds
            )
            self.generate = self.generate_openai

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate_anthropic(self, user_content, temperature=0.3, max_tokens=10000):
        messages = [{"role": "user", "content": user_content}]
        async def _generate():
            messages_response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            return messages_response

        response = asyncio.run(_generate())
        response_text = response.content[0].text

        return response_text

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate_openai(self, user_content, temperature=0.3, max_tokens=10000):
        messages = [{"role": "user", "content": user_content}]
        kwargs = {
            'model': self.model,
            'messages': messages,
            'max_completion_tokens': max_tokens,
        }
        if 'o3-mini' not in self.model: # o3-mini doesn't support temperature
            kwargs['temperature'] = temperature

        response = self.client.chat.completions.create(**kwargs)

        response_text = response.choices[0].message.content
        return response_text

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate_google(self, user_content, temperature=0.3, max_tokens=10000):
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )
        return response.text
