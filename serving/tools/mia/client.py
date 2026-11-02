"""What the checks in tools/ share: the server's URL (API_URL, default http://127.0.0.1:8888, or just PORT), a
one-line error instead of a traceback, and random prose of a given length."""
import os
import random
import sys
import urllib.error
import urllib.request

API_URL = os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888")).rstrip("/")
URL = API_URL + "/v1/chat/completions"


from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import checklib as C


_api_key = C.api_key
# Explicit key-file selection uses C.use_api_key_args/C.add_api_key_arg.
# Importing this module never reads credentials or installs a global opener.

WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine garden theory market winter method bridge letter window voice paper field").split()


def prose(tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    while len(out) < tokens * 0.72:          # ~1.14 tokens a word: the prompt comes out at ~0.82 x tokens
        sentence = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
        out += sentence[:-1] + [sentence[-1] + "."]
    return " ".join(out)


def open_url(req: urllib.request.Request, timeout: float):
    """urlopen, with a one-line message instead of a traceback when the server is not there or refuses."""

    try:
        if C.origin(req.full_url) != C.origin(API_URL):
            raise ValueError("request origin differs from configured API_URL")
        for name, value in C.headers().items():
            req.add_unredirected_header(name, value)
        return C.OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        sys.exit(C.redact(f"the server at {API_URL} answered {exc.code}: {exc.read().decode(errors='replace')[:300]}"))
    except urllib.error.URLError as exc:
        sys.exit(C.redact(f"cannot reach the server at {API_URL} ({exc.reason}): is it running? (./start.sh)"))

