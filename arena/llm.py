from abc import ABC, abstractmethod
from groq import Groq
import google.generativeai as genai
import logging
from typing import Dict, Type, List, Optional
import os
import time
import json
import re
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

MODEL_PROVIDER = {
    "openai/gpt-oss-120b": "groq",
    "openai/gpt-oss-20b": "groq",
    "qwen/qwen3-32b": "groq",
    "qwen/qwen3.6-27b": "groq",
    "llama-3.3-70b-versatile": "groq",
    "meta-llama/llama-4-scout-17b-16e-instruct": "groq",
    "llama-3.1-8b-instant": "groq",
    "groq/compound": "groq",
    "groq/compound-mini": "groq",
    "gemini-2.5-flash": "gemini"
}

class LLMParsingException(Exception):
    pass

def strip_markdown(text: str) -> str:
    """Remove markdown formatting like ```json and ```"""
    text = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
    return text.strip()

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks. Handle truncated blocks where </think> is missing."""
    if "<think>" in text:
        if "</think>" in text:
            # Normal case: strip the complete thinking block
            parts = text.split("</think>")
            return parts[-1].strip()
        else:
            # Truncated: <think> present but </think> never arrived.
            # Try to salvage any JSON that might appear after the thinking text.
            before_think = text.split("<think>")[0].strip()
            if "{" in before_think:
                return before_think
            return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

def extract_json(text: str) -> str:
    """Extract JSON from the text by finding the first { and last }"""
    left = text.find("{")
    right = text.rfind("}")
    if left > -1 and right > -1:
        return text[left:right + 1]
    return text

def clean_response(text: str) -> str:
    """Apply all cleanup methods to get raw JSON string"""
    if not text:
        return "{}"
    text = strip_thinking(text)
    text = strip_markdown(text)
    text = extract_json(text)
    return text.strip()

class LLM(ABC):
    """
    An abstract superclass for interacting with LLMs - subclass for Groq and Gemini
    """

    model_names = []

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 3000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        """
        Send a message
        :param system: the context in which this message is to be taken
        :param user: the prompt
        :param max_tokens: max number of tokens to generate (kept for backward compat)
        :return: the response from the AI
        """
        result = self.protected_send(system, user)
        result = clean_response(result)
        logger.warning(f"[{self.model_name}] Cleaned response: {result[:200]}")
        return result

    def protected_send(self, system: str, user: str) -> str:
        """
        Wrap the send call in an exception handler, giving the LLM 2 chances in total, in case
        of overload errors. If it fails, then it forfeits!
        """
        retries = 2
        while retries:
            retries -= 1
            try:
                start_time = time.time()
                raw = self._send(system, user)
                latency = time.time() - start_time
                logger.warning(f"[{self.model_name}] API call took {latency:.2f}s")
                if not raw:
                    logger.error(f"[{self.model_name}] API returned empty/None content")
                    raw = "{}"
                else:
                    logger.warning(f"[{self.model_name}] Raw response: {raw[:300]}")
                return raw
            except Exception as e:
                logger.error(f"[{self.model_name}] Exception on calling LLM: {e}")
                if retries:
                    logger.warning(f"[{self.model_name}] Waiting 1s and retrying...")
                    time.sleep(1)
        return "{}"

    @abstractmethod
    def _send(self, system: str, user: str) -> str:
        """
        Send a message to the model - must be implemented by subclasses
        """
        pass

    @classmethod
    def all_supported_model_names(cls) -> List[str]:
        """
        Return a list of all the model names supported by all providers.
        """
        return list(MODEL_PROVIDER.keys())

    @classmethod
    def all_model_names(cls) -> List[str]:
        """
        Return a list of all the model names supported.
        Use the ones specified in the MODEL_PROVIDER map, but also check if there's
        an env variable set that restricts the models.
        """
        models = cls.all_supported_model_names()
        allowed = os.getenv("MODELS")
        logger.info(f"Allowed models: {allowed}")
        if allowed:
            allowed_models = allowed.split(",")
            return [m for m in allowed_models if m in models]
        return models

    @classmethod
    def create(cls, model_name: str, temperature: float = 0.0) -> 'LLM':
        """
        Return an instance of a subclass that corresponds to this model_name
        :param model_name: a string to describe this model
        :param temperature: the creativity setting
        :return: a new instance of a subclass of LLM
        """
        provider = MODEL_PROVIDER.get(model_name)
        if not provider:
            raise ValueError(f"Unrecognized LLM model name specified: {model_name}")

        logger.info(f"Creating LLM client for {model_name} via {provider}")
        if provider == "groq":
            return GroqLLM(model_name, temperature)
        elif provider == "gemini":
            return GeminiLLM(model_name, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")

class GroqLLM(LLM):
    """
    A class to act as an interface to the Groq API
    """
    def __init__(self, model_name: str, temperature: float = 0.0):
        super().__init__(model_name, temperature)
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key, timeout=60.0)

    def _send(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        finish_reason = response.choices[0].finish_reason
        logger.warning(f"[{self.model_name}] finish_reason={finish_reason}")
        if finish_reason == "length":
            logger.error(f"[{self.model_name}] Response TRUNCATED (hit max_tokens={self.max_tokens})")
        content = response.choices[0].message.content
        return content or "{}"

class GeminiLLM(LLM):
    """
    A class to act as an interface to the Google Gemini API
    """
    def __init__(self, model_name: str, temperature: float = 0.0):
        super().__init__(model_name, temperature)
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)

    def _send(self, system: str, user: str) -> str:
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json"
            }
        )
        response = model.generate_content(
            user,
            request_options={"timeout": 60.0}
        )
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        logger.warning(f"[{self.model_name}] finish_reason={finish_reason}")
        if finish_reason and str(finish_reason) == "MAX_TOKENS":
            logger.error(f"[{self.model_name}] Response TRUNCATED (hit max_output_tokens={self.max_tokens})")
        return response.text
