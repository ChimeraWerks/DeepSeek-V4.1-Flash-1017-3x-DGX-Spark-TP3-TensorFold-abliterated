"""PDF documents (TensorFold --vision-pdf): counted by the engine as sent, gated and routed as media."""
import asyncio
import json
import unittest

import httpx

from chimera_sparks.config import Config
from chimera_sparks.engine import PDF_PAGE_RESERVE, Engine, GatewayError
from chimera_sparks.laky.media_admission import has_images
from chimera_sparks.laky.vision_jobs import standalone

MODEL = "deepseek-v4.1-flash"
DOCUMENT = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"}}
FILE = {"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,JVBERi0xLjQK"}}
INPUT_FILE = {"type": "input_file", "filename": "a.pdf", "file_data": "JVBERi0xLjQK"}
# An agent client's file read of a PDF: the document arrives in the tool_result
READ = {"model": MODEL, "max_tokens": 64000, "messages": [
    {"role": "user", "content": "summarize the paper"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.pdf"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [DOCUMENT]}]}]}
TEXT_ONLY = {"type": "error", "error": {"type": "invalid_request_error",
             "message": "this server accepts text parts only; image, audio and video inputs are unsupported"}}


def count(body, api, reply):
    seen = []

    def native(request):
        seen.append((request.url.path, json.loads(request.content)))
        return reply(seen[-1][1])

    async def run():
        engine = Engine(Config(), httpx.MockTransport(native))
        try:
            return await engine.count_input(body, api)
        finally:
            await engine.close()
    return asyncio.run(run()), seen


class DocumentCountTests(unittest.TestCase):
    def test_documents_go_to_the_engine_count_as_sent(self):
        result, seen = count(READ, "messages", lambda _: httpx.Response(200, json={"input_tokens": 3200, "image_tokens": 3100}))
        self.assertEqual(result, (3200, "native-exact"))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "/v1/messages/count_tokens")
        self.assertEqual(seen[0][1]["messages"][2]["content"][0]["content"][0], DOCUMENT)
        chat = {"model": MODEL, "messages": [{"role": "user", "content": [{"type": "text", "text": "read"}, FILE]}]}
        result, seen = count(chat, "chat", lambda _: httpx.Response(200, json={"count": 900, "image_tokens": 890}))
        self.assertEqual((result, seen[0][0]), ((900, "native-exact"), "/tokenize"))
        self.assertEqual(seen[0][1]["messages"], chat["messages"])
        responses = {"model": MODEL, "input": [{"role": "user", "content": [INPUT_FILE]}]}
        result, seen = count(responses, "responses", lambda _: httpx.Response(200, json={"count": 700, "image_tokens": 690}))
        self.assertEqual(result, (700, "native-exact"))
        self.assertEqual((seen[0][1]["api"], seen[0][1]["input"]), ("responses", responses["input"]))

    def test_an_engine_that_counts_text_only_reserves_a_documents_page_limit(self):
        def old_engine(sent):
            if '"document"' in json.dumps(sent):
                return httpx.Response(400, json=TEXT_ONLY)
            return httpx.Response(200, json={"input_tokens": 30})
        result, seen = count(READ, "messages", old_engine)
        self.assertEqual(result, (30 + PDF_PAGE_RESERVE * 1024, "native-text+media-reserve"))
        self.assertEqual(seen[1][1]["messages"][2]["content"][0]["content"][0], {"type": "text", "text": "[document]"})

    def test_a_refused_document_keeps_the_engines_words_and_status(self):
        off = "PDF inputs are off on this server; start the server without --no-vision-pdf"
        missing = ("libpdfium.so is required for PDF inputs; install PDFium (chromium/7725) in lib/ "
                   "beside the server or set TENSORFOLD_LIBPDFIUM to its file")
        for status, message in ((400, off), (501, missing), (400, "a PDF may have at most 16 pages (TENSORFOLD_MAX_PDF_PAGES); this request has 20")):
            with self.assertRaises(GatewayError) as caught:
                count(READ, "messages", lambda _: httpx.Response(status, json={"type": "error", "error": {"message": message}}))
            self.assertEqual((caught.exception.status_code, str(caught.exception)), (status, message))


class DocumentMediaTests(unittest.TestCase):
    def test_documents_are_media_for_the_gate_and_standalone_routing(self):
        for part in (DOCUMENT, FILE, INPUT_FILE):
            self.assertTrue(has_images({"messages": [{"role": "user", "content": [part]}]}), part)
        self.assertTrue(has_images(READ))
        self.assertTrue(has_images({"input": [{"role": "user", "content": [INPUT_FILE]}]}))
        # a one-question PDF request is a durable vision job, as a one-question image request is
        self.assertTrue(standalone({"messages": [{"role": "user", "content": [{"type": "text", "text": "summarize"}, FILE]}]}))
        self.assertFalse(standalone(READ))  # an agent's history stays on the ordinary path
        # a tool schema whose property is named "file" is not a document
        schema = {"type": "object", "properties": {"file": {"type": "string"}}}
        self.assertFalse(has_images({"messages": [{"role": "user", "content": "hi"}],
                                     "tools": [{"type": "function", "function": {"name": "f", "parameters": schema}}]}))

    def test_a_document_request_holds_the_media_gate_until_its_stream_closes(self):
        def native(request):
            if request.url.path == "/v1/messages/count_tokens":
                return httpx.Response(200, json={"input_tokens": 3200, "image_tokens": 3100})
            return httpx.Response(200, content=b"event: message_stop\ndata: {}\n\n", headers={"content-type": "text/event-stream"})

        async def run():
            engine = Engine(Config(), httpx.MockTransport(native))
            try:
                body = {**READ, "max_tokens": 64}
                free = engine.media_gate._value
                handle = await engine.begin_stream(body, api="messages")
                held = engine.media_gate._value
                await engine.close_stream(handle)
                return handle.media, handle.prepared["token_method"], free, held, engine.media_gate._value
            finally:
                await engine.close()
        media, method, free, held, after = asyncio.run(run())
        self.assertTrue(media)
        self.assertEqual(method, "native-exact")
        self.assertEqual((held, after), (free - 1, free))


if __name__ == "__main__":
    unittest.main()
