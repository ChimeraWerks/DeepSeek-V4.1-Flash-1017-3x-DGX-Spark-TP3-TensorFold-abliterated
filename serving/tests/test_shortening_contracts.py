"""Focused regressions for shorthand contracts and immutable live turns."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest

from chimera_sparks.config import Config
from chimera_sparks.policy import normalize
from chimera_sparks.shortening import Shortener, CompactionError
from chimera_sparks.laky import responses_checkpoint as cp


class ShorteningContractTests(unittest.IsolatedAsyncioTestCase):
    def cfg(self):
        return replace(Config(), auto_compact=True, compact_trigger_tokens=16000,
            compact_target_tokens=12000, compact_recent_tokens=1,
            compact_chunk_chars=12000, context_tokens=24000,
            max_output_tokens=8192, output_tokens=512).validate()

    def history(self, api, kind='message'):
        messages = [dict(role='system', content='Keep original system instruction.'),
            dict(role='developer', content='Keep developer instruction.'),
            dict(role='user', content='Original task'),
            dict(role='assistant', content='Earlier response ' * 80),
            dict(role='user', content='CURRENT')]
        if api == 'responses':
            for message in messages:
                if kind != 'absent':
                    message['type'] = kind
            for i in range(2):
                messages.extend([
                    dict(type='function_call', call_id=f'live-{i}', name='inspect', arguments='{}'),
                    dict(type='function_call_output', call_id=f'live-{i}', output=f'LIVE-{i} ' * 100)])
        else:
            for i in range(2):
                messages.extend([
                    dict(role='assistant', tool_calls=[dict(id=f'live-{i}', type='function',
                        function=dict(name='inspect', arguments='{}'))]),
                    dict(role='tool', tool_call_id=f'live-{i}', content=f'LIVE-{i} ' * 100)])
        return messages

    async def count(self, body, api):
        return len(json.dumps(body)) // 2, 'focused-offline'

    async def completion(self, system, prompt, material):
        self.materials.append(material)
        result = dict(json.loads(prompt)['checkpoint_control'])
        result.pop('allowed_evidence_ids', None)
        result.pop('required_literals', None)
        for name in cp._LIST_FIELDS:
            result[name] = []
        for name in cp._CLAIM_FIELDS:
            result[name] = dict(text='verified task', evidence_ids=[material.allowed_evidence_ids[0]])
        result['exact_literals'] = []
        result['critical_evidence_ids'] = []
        return result

    async def compact_case(self, api, kind='message', target=None, hard_count=False):
        self.materials = []
        cfg = self.cfg()
        if target is not None:
            cfg = replace(cfg, compact_target_tokens=target)
        key = 'input' if api == 'responses' else 'messages'
        history = self.history(api, kind)
        body = normalize({key: history}, cfg, api=api)
        original = deepcopy(body)
        async def count(body, api):
            if hard_count:
                return 100000, 'over-hard'
            return await self.count(body, api)
        result, compacted = await Shortener(cfg, count, self.completion).compact(body, 20000, api)
        self.assertTrue(compacted)
        self.assertEqual(body, original)
        self.assertEqual(result[key][-5:], history[-5:])
        self.assertEqual(result[key][:2], history[:2])
        self.assertTrue(self.materials)
        for material in self.materials:
            encoded = json.dumps(material.evidence_chunks)
            self.assertNotIn('LIVE-0', encoded)
            self.assertNotIn('LIVE-1', encoded)
        return result

    async def test_two_live_rounds_preserved_with_small_recent_target(self):
        for api in ('chat', 'responses'):
            with self.subTest(api=api):
                await self.compact_case(api)

    async def test_fitting_passes_do_not_consume_live_rounds(self):
        for api in ('chat', 'responses'):
            with self.subTest(api=api):
                await self.compact_case(api, target=1)

    async def test_shorthand_contracts_through_real_material_and_validation(self):
        for kind in ('message', 'absent', None, ''):
            with self.subTest(kind=kind):
                await self.compact_case('responses', kind)
                material = self.materials[-1]
                self.assertIn('Keep original system instruction.', material.task_lens_json)
                self.assertIn('Keep developer instruction.', material.task_lens_json)

    async def test_protected_suffix_over_hard_limit_refuses(self):
        for api in ('chat', 'responses'):
            with self.subTest(api=api):
                with self.assertRaises(CompactionError):
                    await self.compact_case(api, hard_count=True)

    async def test_live_only_tool_rounds_are_never_checkpoint_sources(self):
        for api in ('chat', 'responses'):
            with self.subTest(api=api):
                cfg = self.cfg()
                key = 'input' if api == 'responses' else 'messages'
                body = normalize({key: self.history(api)[4:]}, cfg, api=api)
                self.materials = []
                with self.assertRaises(CompactionError):
                    await Shortener(cfg, self.count, self.completion).compact(body, 20000, api)
                self.assertEqual(self.materials, [])

    def test_nonmessage_role_does_not_become_query(self):
        self.assertFalse(cp.responses_user_item_is_query(
            dict(type='function_call_output', role='user', content='not a query')))
