from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar
import json
import os

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticModel

load_dotenv()

T = TypeVar('T', bound=PydanticModel)

class BaseLLM(ABC):
    """Abstract base class for language model clients."""
    model_name: str
    model_id: str
    
    @abstractmethod
    def generate_structured(self, prompt: str, response_model: Type[T]) -> T:
        """Generate a structured response from the model."""
        pass    


class OpenAIModel(BaseLLM):
    """OpenAI API client implementation."""
    
    def __init__(self, model_name):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.model_id = f"openai:{self.model_name}"
        self.kwargs = {"max_completion_tokens": 2000} if self.model_name.startswith("o") else {"max_tokens": 500}

    
    def generate_structured(self, prompt: str, response_model: Type[T]) -> T:
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.kwargs,
            response_format=response_model
        )
        return response.choices[0].message.parsed


class AnthropicModel(BaseLLM):
    """Anthropic API client implementation."""
    
    def __init__(self, model_name):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name
        self.model_id = f"anthropic:{self.model_name}"

    def generate_structured(self, prompt: str, response_model: Type[T]) -> T:
        schema = response_model.model_json_schema()
        json_prompt = f"""{prompt}

Respond with a JSON object matching this schema:
{json.dumps(schema, indent=2)}

You may wrap the JSON in ```json blocks.
- Do not include any explanatory text before or after the JSON
- Do not provide multiple JSON objects or corrections
- You may wrap the JSON in ```json ``` code blocks. Example response:```json
{{
    "topic": "example topic",
    "option_a": "This is option A",
    "option_b": "This is option B"
}}
```
"""
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": json_prompt}],
            max_tokens=500
        )
        
        response_text = response.content[0].text.strip()
        
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        else:
            json_str = response_text
        
        return response_model.model_validate_json(json_str)


class GoogleGeminiModel(BaseLLM):
    """Google Gemini API client implementation."""
    
    def __init__(self, model_name):
        # The client gets the API key from the environment variable `GEMINI_API_KEY`
        assert os.getenv("GEMINI_API_KEY") is not None, "GEMINI_API_KEY is not set"
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.model_id = f"google:{self.model_name}"
        
        # Check if thinking mode is enabled based on model name suffix
        self.thinking_enabled = model_name.endswith("-thinking")
        
        # If thinking mode, remove the suffix for the actual API call
        if self.thinking_enabled:
            self.api_model_name = model_name[:-9]  # Remove "-thinking"
        else:
            self.api_model_name = model_name
    
    def generate_structured(self, prompt: str, response_model: Type[T]) -> T:
        schema = response_model.model_json_schema()
        json_prompt = f"""{prompt}

Respond with a JSON object matching this schema:
{json.dumps(schema, indent=2)}

You may wrap the JSON in ```json blocks.
- Do not include any explanatory text before or after the JSON
- Do not provide multiple JSON objects or corrections
- You may wrap the JSON in ```json ``` code blocks. Example response:
```json
{{
    "topic": "example topic", 
    "option_a": "This is option A",
    "option_b": "This is option B"
}}
```
"""
        # Configure thinking based on model name
        if self.thinking_enabled:
            thinking_config = genai.types.ThinkingConfig(thinking_budget=-1)  # Dynamic thinking
        else:
            thinking_config = genai.types.ThinkingConfig(thinking_budget=0)   # No thinking

        response = self.client.models.generate_content(
            model=self.api_model_name,
            contents=json_prompt,
            config=genai.types.GenerateContentConfig(
                thinking_config=thinking_config
            ),
        )
        response_text = response.text.strip()
        
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        else:
            json_str = response_text
        
        return response_model.model_validate_json(json_str)


class GrokModel(BaseLLM):
    """Grok API client implementation (uses OpenAI-compatible API)."""
    
    def __init__(self, model_name):
        self.client = OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        self.model_name = model_name
        self.model_id = f"grok:{self.model_name}"
    
    def generate_structured(self, prompt: str, response_model: Type[T]) -> T:
        # Grok doesn't support structured output, so use JSON prompt approach like Anthropic
        schema = response_model.model_json_schema()
        json_prompt = f"""{prompt}

Respond with a JSON object matching this schema:
{json.dumps(schema, indent=2)}

You may wrap the JSON in ```json blocks.
- Do not include any explanatory text before or after the JSON
- Do not provide multiple JSON objects or corrections
- You may wrap the JSON in ```json ``` code blocks. Example response:
```json
{{
    "topic": "example topic",
    "option_a": "This is option A", 
    "option_b": "This is option B"
}}
```
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": json_prompt}],
            # max_tokens=500
        )
        # print(response)
        # assert False
        
        response_text = response.choices[0].message.content.strip()
        
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
        else:
            json_str = response_text
        
        return response_model.model_validate_json(json_str)


def get_model(model_id: str) -> BaseLLM:
    """Factory function to get the appropriate model client."""
    if ":" not in model_id:
        raise ValueError(f"Model ID must be in format 'provider:model'. Got: {model_id}")
    
    provider, model_name = model_id.split(":", 1)
    
    providers = {
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
        "google": GoogleGeminiModel,
        "grok": GrokModel
    }
    
    if provider.lower() not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    model_class = providers[provider.lower()]
    return model_class(model_name)
