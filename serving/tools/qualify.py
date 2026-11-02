"""Short live A/B. Invoke under the shared engine flock and GPU dispatch lease."""
from __future__ import annotations
import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import time
import uuid
import zlib
import httpx


def png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    rows = b"".join(b"\0" + bytes([0, 0, 255]) * 32 for _ in range(32))
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")).decode()


async def stream(client, url, body):
    started = time.monotonic()
    first = last = None
    text = reason = ""
    calls = {}
    usage = {}
    finished = done = False
    async with client.stream("POST", url + "/v1/chat/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                continue
            value = json.loads(data)
            if value.get("error"):
                raise RuntimeError("stream error: " + str(value["error"].get("code", "unknown")))
            usage = value.get("usage") or usage
            for choice in value.get("choices", []):
                delta = choice.get("delta") or {}
                reason += delta.get("reasoning_content") or delta.get("reasoning") or ""
                content = delta.get("content") or ""
                if content:
                    now = time.monotonic()
                    first = first or now
                    last = now
                    text += content
                for call in delta.get("tool_calls") or []:
                    slot = calls.setdefault(call.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    slot["id"] += call.get("id") or ""
                    fn = call.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                    first = first or time.monotonic()
                    last = time.monotonic()
                finished = bool(choice.get("finish_reason")) or finished
    if not done or not finished or not (text.strip() or calls):
        raise RuntimeError("incomplete or empty generation")
    elapsed = time.monotonic() - started
    tokens = usage.get("completion_tokens", 0)
    return {"text": text, "reasoning_characters": len(reason), "reasoning_sha256": hashlib.sha256(reason.encode()).hexdigest(),
            "calls": list(calls.values()), "usage": usage, "tokens": tokens, "wall_seconds": elapsed,
            "ttft_seconds": first - started if first else None,
            "decode_seconds": last - first if first and last and last > first else None,
            "decode_tps": (tokens - 1) / (last - first) if tokens > 1 and first and last and last > first else None}


async def vision(client, url):
    key = "recipe-qualification-" + uuid.uuid4().hex
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Name the solid image color. Reply with only Blue."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png()}}]}],
        "temperature": 0, "top_p": 1, "top_k": 0, "reasoning_effort": "none", "max_tokens": 64}
    accepted = await client.post(url + "/v1/vision/jobs", json=body, headers={"Idempotency-Key": key})
    accepted.raise_for_status()
    identity = accepted.json()["job_id"]
    # This separate request proves accepted work is independent of its submitter.
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        response = await client.get(url + "/v1/vision/jobs/" + identity)
        response.raise_for_status()
        status = response.json()
        if status["status"] in {"succeeded", "failed", "unknown"}:
            break
        await asyncio.sleep(.25)
    else:
        raise RuntimeError("vision deadline; retain job ID " + identity)
    if status["status"] != "succeeded":
        raise RuntimeError("vision state " + status["status"] + "; retain " + identity)
    text = status["result"]["data"]["choices"][0]["message"]["content"]
    if "blue" not in text.lower():
        raise RuntimeError("vision did not identify blue")
    again = await client.post(url + "/v1/vision/jobs", json=body, headers={"Idempotency-Key": key})
    again.raise_for_status()
    if again.json()["job_id"] != identity:
        raise RuntimeError("vision deduplication changed accepted identity")
    return {"job_id": identity, "backend_request_id": status["backend_request_id"], "status": status["status"], "answer": text, "dedup_same_job": True}


async def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"functional": {}, "speed": [], "settings": {"temperature": 0, "top_p": 1, "top_k": 0, "reasoning_effort": "none", "max_tokens": args.tokens}}
    headers = {"Authorization": "Bearer " + os.environ[args.key_env]}
    async with httpx.AsyncClient(headers=headers, timeout=240, trust_env=False, limits=httpx.Limits(max_connections=16)) as client:
        for arm, url in [("ds41f", args.a), ("recipe", args.b)]:
            print("functional " + arm, flush=True)
            base = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Reply with exactly ready."}],
                    "temperature": 0, "top_p": 1, "top_k": 0, "reasoning_effort": "none", "max_tokens": 64}
            answer = await stream(client, url, base)
            if "ready" not in answer["text"].lower():
                raise RuntimeError("stream check failed")
            tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather in a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
            tool = await stream(client, url, {**base, "messages": [{"role": "user", "content": "Get the weather in Boston using the tool."}],
                "tools": tools, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "max_tokens": 192})
            if not tool["calls"] or tool["calls"][0]["name"] != "get_weather":
                raise RuntimeError("required tool check failed")
            json.loads(tool["calls"][0]["arguments"])
            call = tool["calls"][0]
            continuation = await stream(client, url, {**base, "messages": [
                {"role": "user", "content": "Get the weather in Boston using the tool."},
                {"role": "assistant", "content": None, "tool_calls": [{"id": call["id"], "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]}}]},
                {"role": "tool", "tool_call_id": call["id"], "content": "Boston is sunny, 20 C. Tell the user."}], "max_tokens": 96})
            if "sunny" not in continuation["text"].lower():
                raise RuntimeError("tool continuation check failed")
            nonstream = await client.post(url + "/v1/chat/completions", json={**base, "stream": False})
            nonstream.raise_for_status()
            if "ready" not in nonstream.json()["choices"][0]["message"]["content"].lower():
                raise RuntimeError("nonstream final check failed")
            reasoning = await stream(client, url, {**base, "messages": [{"role": "user", "content": "Compute 17 times 19. Think briefly and give the exact answer."}],
                "reasoning_effort": "low", "thinking_budget": 256, "max_tokens": 512})
            if "323" not in reasoning["text"] or not reasoning["reasoning_characters"]:
                raise RuntimeError("reasoning output check failed")
            image = await vision(client, url)
            results["functional"][arm] = {"stream": answer, "tool": tool, "tool_continuation": continuation,
                "nonstream": "passed", "reasoning": reasoning, "vision": image}
            (out/"ab-results.json").write_text(json.dumps(results, indent=2) + "\n")
        for concurrency in [1, 12]:
            for rep, order in enumerate([[("ds41f", args.a), ("recipe", args.b)], [("recipe", args.b), ("ds41f", args.a)]], 1):
                for arm, url in order:
                    print(f"speed {arm} C{concurrency} rep{rep}", flush=True)
                    started = time.monotonic()
                    async def one(i):
                        return await stream(client, url, {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content":
                            f"Trial {rep}, stream {i}. Explain hash table collision handling in detail, with multiple Python examples. Continue with useful details until the output limit."}], **results["settings"], "seed": 12345})
                    rows = await asyncio.gather(*(one(i) for i in range(concurrency)))
                    elapsed = time.monotonic() - started
                    summary = {"arm": arm, "concurrency": concurrency, "rep": rep, "wall_seconds": elapsed,
                               "aggregate_e2e_tps": sum(r["tokens"] for r in rows) / elapsed,
                               "median_decode_tps": statistics.median(r["decode_tps"] for r in rows if r["decode_tps"]),
                               "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in rows), "rows": rows}
                    results["speed"].append(summary)
                    (out/"ab-results.json").write_text(json.dumps(results, indent=2) + "\n")
                    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    print("A/B completed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--key-env", default="SPARK_API_KEY")
    asyncio.run(main(p.parse_args()))
