import json
import sys

# Read the dataset
data = []
with open("final_veomni_training.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Extract text content from assistant messages
texts = []
for entry in data:
    if "messages" in entry:
        for msg in entry["messages"]:
            if msg["role"] == "assistant":
                for content in msg["content"]:
                    if content["type"] == "text":
                        texts.append(content["text"])

# Calculate statistics
total_entries = len(data)
total_chars = sum(len(t) for t in texts)
avg_chars = total_chars / len(texts) if texts else 0

# Token estimation (rough: 1 token ≈ 4 chars for English)
avg_input_tokens = avg_chars / 4
avg_output_tokens = 150  # Estimated Q&A generation output

print(f"{'=' * 70}")
print(f"DATASET STATISTICS")
print(f"{'=' * 70}")
print(f"Total entries:          {total_entries:,}")
print(f"Average chars/entry:    {avg_chars:,.0f}")
print(f"Average input tokens:   {avg_input_tokens:,.0f}")
print(f"Estimated output tokens: {avg_output_tokens}")
print()

print(f"{'=' * 70}")
print(f"API REQUEST ESTIMATES")
print(f"{'=' * 70}")
print(f"Total API calls needed: {total_entries:,} (1 per entry)")
print()

# Cost estimates for different providers
providers = {
    "OpenAI GPT-4o-mini": {
        "input_per_1m": 0.150,
        "output_per_1m": 0.600,
    },
    "OpenAI GPT-4o": {
        "input_per_1m": 2.50,
        "output_per_1m": 10.00,
    },
    "AWS Bedrock Claude 3.5 Haiku": {
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
    },
    "AWS Bedrock Claude 3.5 Sonnet": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
    "AWS Bedrock Llama 3.1 70B": {
        "input_per_1m": 0.99,
        "output_per_1m": 0.99,
    },
    "AWS Bedrock Llama 3.2 90B": {
        "input_per_1m": 1.20,
        "output_per_1m": 1.20,
    },
    "Anthropic Claude 3.5 Haiku": {
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
    },
    "Anthropic Claude 3.5 Sonnet": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
}

print(f"{'=' * 70}")
print(f"COST ESTIMATES (for {total_entries:,} requests)")
print(f"{'=' * 70}")

for provider, pricing in providers.items():
    total_input_tokens = avg_input_tokens * total_entries
    total_output_tokens = avg_output_tokens * total_entries

    input_cost = (total_input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output_per_1m"]
    total_cost = input_cost + output_cost

    print(f"\n{provider}:")
    print(
        f"  Input:  {total_input_tokens / 1_000_000:.2f}M tokens × ${pricing['input_per_1m']}/M = ${input_cost:.2f}"
    )
    print(
        f"  Output: {total_output_tokens / 1_000_000:.2f}M tokens × ${pricing['output_per_1m']}/M = ${output_cost:.2f}"
    )
    print(f"  TOTAL:  ${total_cost:.2f}")

print()
print(f"{'=' * 70}")
print(f"FREE OPTIONS")
print(f"{'=' * 70}")
print(f"• Ollama (Local):      $0.00 - Run Qwen2.5:7B/14B locally")
print(f"• vLLM (Local):        $0.00 - Use your 1000GB VRAM for inference")
print(f"• LM Studio (Local):   $0.00 - Easy GUI for local models")
print(f"• Google Gemini Flash: $0.00 - Free tier: 15 requests/min")
print()
print(
    f"Estimated time with free tier (15 req/min): {total_entries / 15 / 60:.1f} hours"
)
print(
    f"Estimated time with Ollama (local, ~2 req/sec): {total_entries / 2 / 60:.1f} minutes"
)
