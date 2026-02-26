# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AWS Bedrock client for NeMo Curator.
Supports Claude, Llama, and other Bedrock models.
Includes OpenAI-compatible API support.
"""

import asyncio
import json
import os
from collections.abc import Iterable
from typing import Any

import boto3
from loguru import logger

from nemo_curator.models.client.llm_client import (
    AsyncLLMClient,
    ConversationFormatter,
    GenerationConfig,
    LLMClient,
)


class BedrockClient(LLMClient):
    """
    Synchronous AWS Bedrock client for NeMo Curator.

    Usage:
        client = BedrockClient(
            region_name="us-east-1",
            aws_access_key_id="...",  # Optional if using AWS CLI config
            aws_secret_access_key="...",  # Optional if using AWS CLI config
        )
    """

    def __init__(
        self,
        region_name: str = "us-west-2",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        """
        Initialize Bedrock client.

        Args:
            region_name: AWS region (default: us-west-2)
            aws_access_key_id: AWS access key (optional, uses AWS_ACCESS_KEY_ID env var or default credentials)
            aws_secret_access_key: AWS secret key (optional, uses AWS_SECRET_ACCESS_KEY env var)
            aws_session_token: AWS session token (optional, uses AWS_TOKEN or AWS_SESSION_TOKEN env var)
            profile_name: AWS profile name (optional, from ~/.aws/credentials)
        """
        import os

        self.region_name = region_name
        # Read from environment variables if not provided
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        # Support both AWS_TOKEN and AWS_SESSION_TOKEN
        self.aws_session_token = (
            aws_session_token
            or os.getenv("AWS_TOKEN")
            or os.getenv("AWS_SESSION_TOKEN")
        )
        self.profile_name = profile_name
        self.client = None

    def setup(self) -> None:
        """Setup the Bedrock client."""
        session_kwargs = {"region_name": self.region_name}

        if self.profile_name:
            session_kwargs["profile_name"] = self.profile_name

        session = boto3.Session(**session_kwargs)

        client_kwargs = {}
        if self.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            client_kwargs["aws_session_token"] = self.aws_session_token

        self.client = session.client("bedrock-runtime", **client_kwargs)
        logger.info(f"Bedrock client initialized in region {self.region_name}")

    def _format_messages_for_bedrock(
        self, messages: list[dict], model: str
    ) -> tuple[str | None, list[dict]]:
        """
        Format messages for Bedrock API.
        Returns (system_prompt, conversation_messages)
        """
        system_prompt = None
        conversation = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                conversation.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                        if isinstance(msg["content"], str)
                        else json.dumps(msg["content"]),
                    }
                )

        return system_prompt, conversation

    def _build_request_body(
        self,
        messages: list[dict],
        model: str,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        """Build request body for specific Bedrock model."""
        system_prompt, conversation = self._format_messages_for_bedrock(messages, model)

        # Google Gemma models
        if "google.gemma" in model:
            # Format as Gemma chat template
            prompt = ""
            if system_prompt:
                prompt += f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"
            for msg in conversation:
                prompt += (
                    f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"
                )
            prompt += "<start_of_turn>model\n"

            body = {
                "prompt": prompt,
                "max_tokens": generation_config.max_tokens or 2048,
                "temperature": generation_config.temperature or 0.7,
                "top_p": generation_config.top_p or 0.9,
                "top_k": generation_config.top_k or 40,
            }
            if generation_config.stop:
                body["stop_sequences"] = (
                    generation_config.stop
                    if isinstance(generation_config.stop, list)
                    else [generation_config.stop]
                )

        # Claude models (Anthropic)
        elif "anthropic.claude" in model:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": conversation,
                "max_tokens": generation_config.max_tokens or 2048,
                "temperature": generation_config.temperature or 0.0,
                "top_p": generation_config.top_p or 0.95,
            }
            if system_prompt:
                body["system"] = system_prompt
            if generation_config.stop:
                body["stop_sequences"] = (
                    generation_config.stop
                    if isinstance(generation_config.stop, list)
                    else [generation_config.stop]
                )

        # Meta Llama models
        elif "meta.llama" in model:
            # Format as prompt for Llama
            prompt = ""
            if system_prompt:
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            for msg in conversation:
                prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

            body = {
                "prompt": prompt,
                "max_gen_len": generation_config.max_tokens or 2048,
                "temperature": generation_config.temperature or 0.0,
                "top_p": generation_config.top_p or 0.95,
            }

        # Amazon Titan models
        elif "amazon.titan" in model:
            # Combine system and user messages
            prompt = ""
            if system_prompt:
                prompt += f"{system_prompt}\n\n"
            for msg in conversation:
                prompt += f"{msg['content']}\n"

            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": generation_config.max_tokens or 2048,
                    "temperature": generation_config.temperature or 0.0,
                    "topP": generation_config.top_p or 0.95,
                },
            }
            if generation_config.stop:
                body["textGenerationConfig"]["stopSequences"] = (
                    generation_config.stop
                    if isinstance(generation_config.stop, list)
                    else [generation_config.stop]
                )

        else:
            raise ValueError(f"Unsupported Bedrock model: {model}")

        return body

    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        """Query Bedrock model."""
        if self.client is None:
            raise RuntimeError("Client not initialized. Call setup() first.")

        # Use default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)

        messages_list = list(messages)
        body = self._build_request_body(messages_list, model, generation_config)

        try:
            response = self.client.invoke_model(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            # Parse response based on model type
            if "google.gemma" in model:
                # Gemma returns text in "completion" field
                return [response_body.get("completion", response_body.get("text", ""))]
            elif "anthropic.claude" in model:
                content = response_body["content"]
                if isinstance(content, list):
                    return [
                        item["text"] for item in content if item.get("type") == "text"
                    ]
                return [content]
            elif "meta.llama" in model:
                return [response_body["generation"]]
            elif "amazon.titan" in model:
                return [result["outputText"] for result in response_body["results"]]
            else:
                raise ValueError(f"Unknown response format for model: {model}")

        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            raise


class AsyncBedrockClient(AsyncLLMClient):
    """
    Asynchronous AWS Bedrock client for NeMo Curator.

    Usage:
        client = AsyncBedrockClient(
            region_name="us-east-1",
            max_concurrent_requests=10,  # Control concurrency
            max_retries=3,
            base_delay=1.0,
        )
    """

    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        max_concurrent_requests: int = 10,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        """
        Initialize async Bedrock client.

        Args:
            region_name: AWS region (default: us-east-1)
            aws_access_key_id: AWS access key (optional)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional)
            profile_name: AWS profile name (optional)
            max_concurrent_requests: Maximum concurrent API calls
            max_retries: Number of retries on failure
            base_delay: Base delay for exponential backoff
        """
        super().__init__(max_concurrent_requests, max_retries, base_delay)
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.profile_name = profile_name
        self.sync_client = None

    def setup(self) -> None:
        """Setup the async Bedrock client."""
        # Use the sync client for actual API calls (boto3 doesn't have native async support)
        self.sync_client = BedrockClient(
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            profile_name=self.profile_name,
        )
        self.sync_client.setup()
        logger.info(
            f"Async Bedrock client initialized with max_concurrent_requests={self.max_concurrent_requests}"
        )

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        """
        Internal implementation of query_model.
        Uses asyncio.to_thread to run sync boto3 calls without blocking.
        """
        # Run the synchronous boto3 call in a thread pool
        return await asyncio.to_thread(
            self.sync_client.query_model,
            messages=messages,
            model=model,
            conversation_formatter=conversation_formatter,
            generation_config=generation_config,
        )


class AsyncBedrockOpenAIClient(AsyncLLMClient):
    """
    Asynchronous AWS Bedrock client using OpenAI-compatible API for NeMo Curator.

    This client uses the OpenAI-compatible endpoint for AWS Bedrock, which simplifies
    the implementation and doesn't require boto3.

    Usage:
        client = AsyncBedrockOpenAIClient(
            base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
            api_key=os.environ["AWS_SESSION_TOKEN"],
            max_concurrent_requests=10,
        )
    """

    def __init__(
        self,
        base_url: str = "https://bedrock-mantle.us-west-2.api.aws/v1",
        api_key: str | None = None,
        max_concurrent_requests: int = 10,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        """
        Initialize async Bedrock OpenAI-compatible client.

        Args:
            base_url: Bedrock OpenAI-compatible endpoint (default: us-west-2)
            api_key: AWS session token (uses AWS_SESSION_TOKEN or AWS_TOKEN env var if not provided)
            max_concurrent_requests: Maximum concurrent API calls
            max_retries: Number of retries on failure
            base_delay: Base delay for exponential backoff
        """
        super().__init__(max_concurrent_requests, max_retries, base_delay)
        self.base_url = base_url
        self.api_key = (
            api_key or os.getenv("AWS_SESSION_TOKEN") or os.getenv("AWS_TOKEN")
        )
        self.client = None

    def setup(self) -> None:
        """Setup the async OpenAI client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for AsyncBedrockOpenAIClient. "
                "Install it with: pip install openai"
            )

        if not self.api_key:
            raise ValueError(
                "API key is required. Set AWS_SESSION_TOKEN or AWS_TOKEN environment variable, "
                "or pass api_key parameter."
            )

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        logger.info(
            f"Async Bedrock OpenAI client initialized with base_url={self.base_url}, "
            f"max_concurrent_requests={self.max_concurrent_requests}"
        )

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | dict | None = None,
    ) -> list[str]:
        """
        Internal implementation of query_model using OpenAI-compatible API.
        """
        if self.client is None:
            raise RuntimeError("Client not initialized. Call setup() first.")

        # Use default config if none provided
        if generation_config is None:
            generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)

        # Convert messages to list and ensure proper format
        messages_list = list(messages)

        # Build request parameters
        request_params = {
            "model": model,
            "messages": messages_list,
            "max_tokens": generation_config.max_tokens or 2048,
            "temperature": generation_config.temperature or 0.7,
        }

        if generation_config.top_p is not None:
            request_params["top_p"] = generation_config.top_p

        if generation_config.stop:
            request_params["stop"] = (
                generation_config.stop
                if isinstance(generation_config.stop, list)
                else [generation_config.stop]
            )

        try:
            response = await self.client.chat.completions.create(**request_params)

            # Extract text from response
            results = []
            for choice in response.choices:
                if choice.message.content:
                    results.append(choice.message.content)

            return results if results else [""]

        except Exception as e:
            logger.error(f"Bedrock OpenAI API error: {e}")
            raise
