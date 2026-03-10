#!/usr/bin/env python3
"""
SGLang Proxy Server that uses native SGLang Python API.
Provides field filtering and max_tokens adjustment for compatibility.
"""

import asyncio
import json
from aiohttp import web
import argparse
import sglang as sgl
from sglang import RuntimeEndpoint


class SGLangProxy:
    def __init__(self, backend_url="http://localhost:30000", max_context_len=131072):
        self.backend_url = backend_url
        self.max_context_len = max_context_len
        self.runtime = None

    async def init_runtime(self):
        """Initialize SGLang runtime connection"""
        try:
            # Connect to SGLang runtime
            self.runtime = RuntimeEndpoint(self.backend_url)
            print(f"[PROXY] Connected to SGLang runtime at {self.backend_url}")
        except Exception as e:
            print(f"[PROXY ERROR] Failed to connect to SGLang runtime: {e}")
            raise

    def adjust_max_tokens(self, payload, estimated_input_tokens):
        """
        Adjust max_tokens to fit within context window.
        Returns (adjusted_payload, error_message) tuple.
        """
        if "max_tokens" in payload:
            requested_max_tokens = payload["max_tokens"]
            # Calculate available tokens for output
            available_for_output = self.max_context_len - estimated_input_tokens

            # Check if input is already too large
            if available_for_output <= 0:
                error_msg = (
                    f"Input is too large: estimated {estimated_input_tokens} tokens, "
                    f"but max context is {self.max_context_len} tokens. "
                    f"Please reduce input size or use a model with larger context window."
                )
                print(f"[PROXY ERROR] {error_msg}")
                return None, error_msg

            # Ensure we have at least some tokens for output (minimum 256)
            if available_for_output < 256:
                error_msg = (
                    f"Input is too large: estimated {estimated_input_tokens} tokens, "
                    f"only {available_for_output} tokens available for output. "
                    f"Context limit is {self.max_context_len} tokens. "
                    f"Please reduce input size."
                )
                print(f"[PROXY ERROR] {error_msg}")
                return None, error_msg

            if requested_max_tokens > available_for_output:
                print(
                    f"[PROXY] Adjusting max_tokens from {requested_max_tokens} to {available_for_output}"
                )
                payload["max_tokens"] = available_for_output

        return payload, None

    def count_tokens(self, messages):
        """Estimate token count from messages"""
        # Simple estimation: 1 token ~= 4 characters
        # SGLang doesn't have a direct tokenize method in the Python API
        messages_text = json.dumps(messages)
        estimated = len(messages_text) // 4
        print(f"[PROXY] Estimated token count: {estimated}")
        return estimated

    async def handle_chat_completion(self, request):
        """Proxy /v1/chat/completions using SGLang Python API"""
        try:
            # Parse request body
            body = await request.json()

            # Get token count
            messages = body.get("messages", [])
            estimated_tokens = self.count_tokens(messages)

            # Adjust max_tokens if needed
            body, error_msg = self.adjust_max_tokens(body, estimated_tokens)

            # If input is too large, return error
            if error_msg:
                return web.json_response(
                    {
                        "error": {
                            "message": error_msg,
                            "type": "invalid_request_error",
                            "code": "context_length_exceeded",
                        }
                    },
                    status=400,
                )

            # Filter to known SGLang parameters
            allowed_fields = {
                "messages",
                "model",
                "max_tokens",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "n",
                "stream",
                "stop",
                "logprobs",
                "top_logprobs",
                "user",
            }

            # Log removed fields for debugging
            removed_fields = set(body.keys()) - allowed_fields
            if removed_fields:
                print(f"[PROXY] Removing unsupported fields: {removed_fields}")

            # Filter to only allowed fields
            filtered_body = {k: v for k, v in body.items() if k in allowed_fields}

            # Use SGLang Python API to generate
            try:
                # Build generation parameters
                sampling_params = {}
                if "temperature" in filtered_body:
                    sampling_params["temperature"] = filtered_body["temperature"]
                if "top_p" in filtered_body:
                    sampling_params["top_p"] = filtered_body["top_p"]
                if "max_tokens" in filtered_body:
                    sampling_params["max_new_tokens"] = filtered_body["max_tokens"]
                if "frequency_penalty" in filtered_body:
                    sampling_params["frequency_penalty"] = filtered_body[
                        "frequency_penalty"
                    ]
                if "presence_penalty" in filtered_body:
                    sampling_params["presence_penalty"] = filtered_body[
                        "presence_penalty"
                    ]
                if "stop" in filtered_body:
                    sampling_params["stop"] = filtered_body["stop"]

                # Format messages for SGLang
                # Convert OpenAI format to SGLang format
                prompt_parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    if isinstance(content, str):
                        prompt_parts.append(f"{role}: {content}")
                    elif isinstance(content, list):
                        # Handle multimodal content (text + images)
                        for item in content:
                            if item.get("type") == "text":
                                prompt_parts.append(f"{role}: {item.get('text', '')}")
                            elif item.get("type") == "image_url":
                                # SGLang handles images differently
                                prompt_parts.append(f"{role}: [image]")

                prompt = "\n".join(prompt_parts)

                # Generate using SGLang runtime
                response = await asyncio.to_thread(
                    self.runtime.generate, prompt, sampling_params=sampling_params
                )

                # Convert SGLang response to OpenAI format
                openai_response = {
                    "id": f"chatcmpl-{response.get('meta_info', {}).get('id', 'unknown')}",
                    "object": "chat.completion",
                    "created": response.get("meta_info", {}).get("created", 0),
                    "model": filtered_body.get("model", "unknown"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response.get("text", ""),
                            },
                            "finish_reason": response.get("meta_info", {}).get(
                                "finish_reason", "stop"
                            ),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": response.get("meta_info", {}).get(
                            "prompt_tokens", 0
                        ),
                        "completion_tokens": response.get("meta_info", {}).get(
                            "completion_tokens", 0
                        ),
                        "total_tokens": response.get("meta_info", {}).get(
                            "total_tokens", 0
                        ),
                    },
                }

                return web.json_response(openai_response)

            except Exception as e:
                print(f"[PROXY ERROR] SGLang generation failed: {e}")
                import traceback

                traceback.print_exc()
                return web.json_response(
                    {"error": {"message": str(e), "type": "runtime_error"}}, status=500
                )

        except Exception as e:
            print(f"[ERROR] Request handling failed: {e}")
            import traceback

            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(self, request):
        """Health check endpoint"""
        return web.json_response({"status": "ok"})


async def create_app(proxy):
    await proxy.init_runtime()

    # Set max client request size to 10MB for large images
    app = web.Application(client_max_size=10 * 1024 * 1024)
    app["proxy"] = proxy

    # Health check
    app.router.add_get("/health", proxy.handle_health)
    app.router.add_get("/v1/health", proxy.handle_health)

    # Chat completions endpoint
    app.router.add_post("/v1/chat/completions", proxy.handle_chat_completion)

    return app


def main():
    parser = argparse.ArgumentParser(
        description="SGLang Proxy Server using native Python API"
    )
    parser.add_argument("--port", type=int, default=8000, help="Proxy server port")
    parser.add_argument(
        "--backend-port", type=int, default=30000, help="SGLang backend port"
    )
    parser.add_argument(
        "--max-context", type=int, default=131072, help="Model context length"
    )
    args = parser.parse_args()

    backend_url = f"http://localhost:{args.backend_port}"
    proxy = SGLangProxy(backend_url=backend_url, max_context_len=args.max_context)

    print(f"[PROXY] Starting SGLang proxy on port {args.port}")
    print(f"[PROXY] Backend: {backend_url}")
    print(f"[PROXY] Max context length: {args.max_context}")
    print(f"[PROXY] Using native SGLang Python API")

    web.run_app(create_app(proxy), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
