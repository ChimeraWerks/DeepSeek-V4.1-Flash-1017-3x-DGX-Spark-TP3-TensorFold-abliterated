#!/usr/bin/env python3
"""token_map.py TOKENIZER_JSON [OUT]: rebuild the kit's engram-token-map.json from a tokenizer.json.

Token id -> compressed id, where tokens that normalize alike share one id, as DeepSeek's Engram builds it
(compressed_token_map in TensorFold's deepseek_v41/ops.py). Needs the `tokenizers` package, e.g.
`uv run --no-project --with tokenizers python3 -I tools/assets/token_map.py <MODEL_DIR>/tokenizer.json`.
Prints the vocabulary sizes and the SHA-256; the published tokenizer gives
de40bcdd71f24430ba82752ef385093be962adbf845a1c97db39161597d5dc28.
"""
import hashlib
import json
import sys

from tokenizers import Regex, Tokenizer, normalizers


def token_map(path):
    tok = Tokenizer.from_file(str(path))
    sentinel = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    keys, lookup = {}, []
    for tid in range(tok.get_vocab_size(with_added_tokens=True)):
        text = tok.decode([tid], skip_special_tokens=False)
        key = tok.id_to_token(tid) if "�" in text else (norm.normalize_str(text) or text)
        lookup.append(keys.setdefault(key, len(keys)))
    return lookup, len(keys)


def main(argv):
    if len(argv) not in (1, 2):
        sys.exit(__doc__)
    lookup, compressed = token_map(argv[0])
    data = json.dumps(lookup).encode()
    if len(argv) == 2:
        with open(argv[1], "xb") as f:
            f.write(data)
    print(json.dumps({"tokens": len(lookup), "compressed": compressed, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main(sys.argv[1:])
