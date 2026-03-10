#!/usr/bin/env python3
"""
SGLang Proxy Server that automatically adjusts max_tokens to fit within context window.
Similar to vLLM proxy but adapted for SGLang's API.
"""

import asyncio
import json
from aiohttp import web, ClientSession
import argparse


class SGLangProxy:
    def __init__(self, backend_url="http://localhost:30000", max_context_len=131072):
        self.backend_url = backend_url
        self.max_context_len = max_context_len
        self.session = None

    async def init_session(self):
        self.session = ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

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

    async def count_tokens(self, messages):
        """Use SGLang's tokenize endpoint to get accurate token count"""
        try:
            # SGLang tokenize endpoint
            async with self.session.post(
                f"{self.backend_url}/tokenize",
                json={"text": json.dumps(messages)},
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # SGLang returns {"count": N}
                    token_count = data.get("count", 0)
                    print(f"[PROXY] Accurate token count: {token_count}")
                    return token_count
                else:
                    # Fallback to rough estimation
                    messages_text = json.dumps(messages)
                    estimated = len(messages_text) // 4
                    print(f"[PROXY] Tokenize failed, using estimate: {estimated}")
                    return estimated
        except Exception as e:
            # Fallback to rough estimation on error
            messages_text = json.dumps(messages)
            estimated = len(messages_text) // 4
            print(f"[PROXY] Tokenize error ({e}), using estimate: {estimated}")
            return estimated

    async def handle_chat_completion(self, request):
        """Proxy /v1/chat/completions with automatic max_tokens adjustment"""
        try:
            # Parse request body
            body = await request.json()

            # Get accurate token count
            messages = body.get("messages", [])
            estimated_tokens = await self.count_tokens(messages)

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

            # Remove unsupported fields that might cause warnings
            body.pop("thinking", None)

            # SGLang-specific: Remove fields that might cause 400 errors
            # Keep only known SGLang parameters
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

            # Forward to backend
            async with self.session.post(
                f"{self.backend_url}/v1/chat/completions",
                json=filtered_body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                response_data = await resp.read()

                # Log error responses for debugging
                if resp.status != 200:
                    print(
                        f"[PROXY ERROR] Backend returned {resp.status}: {response_data.decode()}"
                    )

                return web.Response(
                    body=response_data,
                    status=resp.status,
                    content_type="application/json",
                )

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback

            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    async def proxy_request(self, request):
        """Generic proxy for other endpoints"""
        path = request.path

        try:
            async with self.session.request(
                method=request.method,
                url=f"{self.backend_url}{path}",
                data=await request.read(),
                headers={
                    k: v for k, v in request.headers.items() if k.lower() != "host"
                },
            ) as resp:
                response_data = await resp.read()
                return web.Response(
                    body=response_data, status=resp.status, headers=resp.headers
                )
        except Exception as e:
            print(f"[ERROR] Proxy failed: {e}")
            return web.json_response({"error": str(e)}, status=500)


async def create_app(proxy):
    await proxy.init_session()

    # Set max client request size to 10MB for large images
    app = web.Application(client_max_size=10 * 1024 * 1024)
    app["proxy"] = proxy

    # Special handling for chat completions
    app.router.add_post("/v1/chat/completions", proxy.handle_chat_completion)

    # Proxy all other requests
    app.router.add_route("*", "/{path:.*}", proxy.proxy_request)

    async def cleanup(app):
        await proxy.close_session()

    app.on_cleanup.append(cleanup)

    return app


def main():
    parser = argparse.ArgumentParser(description="SGLang Proxy Server")
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

    web.run_app(create_app(proxy), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
