"""Opt-in task-aware shortening using LakyFx 1.2.6 evidence contracts.

Every model pass is explicit, bounded and admitted by the serving gate. No
background generation, cross-request cache, free-form fallback or blind retry.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

from .laky.cache import TTLCache
from .laky.chat_checkpoint import evidence_items
from .laky.compaction import ContextCompactor, ResponsesContextCompactor, CompactionError
from .laky import responses_checkpoint as cp


class Shortener:
    def __init__(self, cfg, count, completion):
        self.cfg, self.count, self.completion = cfg, count, completion
        self.remaining_calls = cfg.checkpoint_max_chunks + 1
        self.settings = SimpleNamespace(
            backend_family="tensorfold", default_max_tokens=cfg.output_tokens,
            compact_trigger_tokens=cfg.compact_trigger_tokens,
            compact_target_tokens=cfg.compact_target_tokens,
            compact_recent_tokens=cfg.compact_recent_tokens,
            compact_chunk_chars=cfg.compact_chunk_chars,
            responses_checkpoint_mode="enforce",
            responses_checkpoint_shadow_sample_rate=0.0,
        )

    @staticmethod
    def chat_body(body):
        return {"model": body.get("model"), "tools": deepcopy(body.get("tools") or []),
                "instructions": json.dumps([m for m in body["messages"]
                    if m.get("role") in {"system", "developer"}], ensure_ascii=False),
                "input": evidence_items(body["messages"])}

    def material(self, body, old, user):
        material = cp.build_checkpoint_material(body=body, old_items=old, current_user=user,
            max_chunk_chars=self.cfg.compact_chunk_chars, max_task_chars=self.cfg.compact_chunk_chars,
            model=self.cfg.model)
        if not material.serving_evidence_complete or not material.evidence_chunks:
            raise CompactionError("checkpoint evidence ledger is incomplete or empty")
        if len(material.evidence_chunks) > self.cfg.checkpoint_max_chunks:
            raise CompactionError("checkpoint history exceeds the configured work bound")
        return material

    def plan(self, body, old, user):
        item = cp.checkpoint_planning_item(self.material(body, old, user))
        return self.text(item), self.cfg.checkpoint_output_tokens

    @staticmethod
    def text(item):
        value = item["content"]
        return value if isinstance(value, str) else "\n".join(p.get("text", "") for p in value)

    async def complete(self, system, prompt, material):
        if self.remaining_calls <= 0:
            raise CompactionError("checkpoint generation exceeded its request work bound")
        self.remaining_calls -= 1
        return await self.completion(system, prompt, material)

    async def generate(self, body, old, user):
        material = self.material(body, old, user)
        partials = []
        for i, chunk in enumerate(material.evidence_chunks, 1):
            value = await self.complete(cp.CHECKPOINT_EXTRACTOR_SYSTEM,
                cp.extraction_prompt(material, chunk, i, len(material.evidence_chunks)), material)
            partials.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        checkpoint = await self.complete(cp.CHECKPOINT_RECONCILER_SYSTEM,
            cp.reconciliation_prompt(material, partials), material)
        checkpoint = cp.bind_checkpoint_exact_literals(checkpoint, material)
        observation = cp.validate_checkpoint(checkpoint, material, min_literal_recall=1.0)
        if not observation.valid:
            raise CompactionError("checkpoint failed its evidence gate: " + ",".join(observation.errors))
        return cp.checkpoint_serving_item(observation)

    async def compact(self, body, count, api, *, force=False):
        cfg = self.cfg
        cap = body["max_output_tokens" if api == "responses" else "max_tokens"]
        hard = cfg.context_tokens - cap - cfg.safety_tokens
        target = min(hard, cfg.compact_target_tokens)
        # Instances/caches belong to this invocation; client histories never mix.
        cache = TTLCache(max_entries=1, ttl_seconds=1)
        async def no_summary(*args):
            raise CompactionError("unvalidated summary is disabled")
        async def count_candidate(candidate):
            return (await self.count(candidate, api))[0]
        async def checkpoint_chat(original, old, user):
            item = await self.generate(self.chat_body(original), evidence_items(old), evidence_items([user])[0])
            return self.text(item)
        def plan_chat(original, old, user):
            return self.plan(self.chat_body(original), evidence_items(old), evidence_items([user])[0])
        async with asyncio.timeout(cfg.checkpoint_timeout):
            if api == "chat":
                compactor = ContextCompactor(settings=self.settings, token_counter=count_candidate,
                    summary_generator=no_summary, cache=cache, checkpoint_generator=checkpoint_chat,
                    checkpoint_planner=plan_chat)
                result = await compactor.maybe_compact(body, count, hard_input_limit=hard,
                    target_input_limit=target, force=force)
                return result.body, result.compacted
            if body.get("previous_response_id"):
                raise CompactionError("shortening requires a full stateless Responses history")
            compactor = ResponsesContextCompactor(settings=self.settings, token_counter=count_candidate,
                summary_generator=no_summary, cache=cache, checkpoint_planner=self.plan)
            result = await compactor.maybe_compact(body, count, hard_input_limit=hard,
                target_input_limit=target, force=force)
            if not result.compacted:
                return result.body, False
            job = result.checkpoint_job
            if job is None:
                raise CompactionError("validated checkpoint source is unavailable")
            item = await self.generate(job.body, job.old_items, job.current_user)
            candidate = deepcopy(result.body)
            candidate["input"][result.checkpoint_index] = item
            if (await self.count(candidate, api))[0] > hard:
                raise CompactionError("validated checkpoint and protected tail exceed context")
            return candidate, True
