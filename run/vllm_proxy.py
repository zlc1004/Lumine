#!/usr/bin/env python3
"""
vLLM Proxy Server that automatically adjusts max_tokens to fit within context window.
This allows clients to send max_tokens=65535 without validation errors.
"""

import asyncio
import json
from aiohttp import web, ClientSession
import argparse


class VLLMProxy:
    def __init__(self, backend_url="http://localhost:8001", max_context_len=65536):
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
        Leaves a safety margin for the input prompt.
        """
        if "max_tokens" in payload:
            requested_max_tokens = payload["max_tokens"]
            # Reserve at least 1024 tokens for input, adjust if needed
            available_for_output = self.max_context_len - max(
                estimated_input_tokens, 1024
            )

            if requested_max_tokens > available_for_output:
                print(
                    f"[PROXY] Adjusting max_tokens from {requested_max_tokens} to {available_for_output}"
                )
                payload["max_tokens"] = available_for_output

        return payload

    async def handle_chat_completion(self, request):
        """Proxy /v1/chat/completions with automatic max_tokens adjustment"""
        try:
            # Parse request body
            body = await request.json()

            # Estimate input token count (rough approximation: 1 token ~= 4 chars)
            messages_text = json.dumps(body.get("messages", []))
            estimated_tokens = len(messages_text) // 4

            # Adjust max_tokens if needed
            body = self.adjust_max_tokens(body, estimated_tokens)

            # Remove unsupported fields that cause warnings
            body.pop("thinking", None)

            # Forward to backend
            async with self.session.post(
                f"{self.backend_url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                response_data = await resp.read()
                return web.Response(
                    body=response_data,
                    status=resp.status,
                    content_type="application/json",
                )

        except Exception as e:
            print(f"[ERROR] {e}")
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

    app = web.Application()
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
    parser = argparse.ArgumentParser(description="vLLM Proxy Server")
    parser.add_argument("--port", type=int, default=8000, help="Proxy server port")
    parser.add_argument(
        "--backend-port", type=int, default=8001, help="vLLM backend port"
    )
    parser.add_argument(
        "--max-context", type=int, default=65536, help="Model context length"
    )
    args = parser.parse_args()

    backend_url = f"http://localhost:{args.backend_port}"
    proxy = VLLMProxy(backend_url=backend_url, max_context_len=args.max_context)

    print(f"[PROXY] Starting vLLM proxy on port {args.port}")
    print(f"[PROXY] Backend: {backend_url}")
    print(f"[PROXY] Max context length: {args.max_context}")

    web.run_app(create_app(proxy), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
