"""A saved, lossless code-point vocabulary; no spelling or Unicode normalization."""

from __future__ import annotations

import json
import string
from pathlib import Path

POLICY = {
    "version": 2,
    "unicode_normalization": "none",
    "order": "logical Unicode code points; never reverse RTL text",
    "diacritics": "preserve all literal code points",
    "whitespace": "preserve internal whitespace including LF/CR; remove at most one terminal LF or CRLF from label files",
    "unknown": "error",
    "sequence": "characters then EOS; repeat EOS to fixed capacity",
}
# Declared inventory, independent of validation/test labels. Additional code points
# come ONLY from training text or an explicitly supplied inventory file.
ARABIC_INVENTORY = (
    string.ascii_letters + string.digits + string.punctuation + " "
    + "،؛؟٪٫٬" + "".join(chr(i) for i in range(0x621, 0x63B))
    + "".join(chr(i) for i in range(0x640, 0x660))
    + "".join(chr(i) for i in range(0x660, 0x66A))
    + "".join(chr(i) for i in range(0x6F0, 0x6FA)) + "ٰٱپچژگکھیے"
)

# A derived training policy, separate from the immutable tokenizer/file policy.
HARAKAT_V1 = frozenset(chr(i) for i in range(0x64B, 0x653))


def training_text(text: str, policy: str = "literal") -> str:
    if policy == "literal":
        return text
    if policy == "no_harakat_v1":
        return "".join(c for c in text if c not in HARAKAT_V1)
    raise ValueError("Unknown training text policy")


def read_literal(path: Path) -> str:
    text = path.read_bytes().decode("utf-8", errors="strict")
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    return text


class CharacterTokenizer:
    PAD, MASK, EOS = 0, 1, 2
    SPECIAL = ["<PAD>", "<MASK>", "<EOS>"]

    def __init__(self, characters: list[str]):
        if any(len(c) != 1 for c in characters):
            raise ValueError("Vocabulary entries must be single code points")
        if len(set(characters)) != len(characters):
            raise ValueError("Duplicate vocabulary characters")
        self.characters = list(characters)
        self.tokens = self.SPECIAL + self.characters
        self.ids = {c: i for i, c in enumerate(self.tokens)}

    @classmethod
    def build(cls, texts, extra_inventory: str = ""):
        chars = set(ARABIC_INVENTORY + extra_inventory)
        for text in texts:
            chars.update(text)
        return cls(sorted(chars, key=ord))

    def __len__(self):
        return len(self.tokens)

    def encode(self, text: str, capacity: int, *, eos_fill: bool = True) -> list[int]:
        if len(text) + 1 > capacity:
            raise ValueError(f"Label overflow: {len(text)} characters + EOS > capacity {capacity}")
        unknown = sorted(set(text) - set(self.characters), key=ord)
        if unknown:
            raise ValueError("Out-of-vocabulary characters: " + ", ".join(f"{c!r} U+{ord(c):04X}" for c in unknown))
        ids = [self.ids[c] for c in text] + [self.EOS]
        return ids + [self.EOS if eos_fill else self.PAD] * (capacity - len(ids))

    def decode(self, ids) -> tuple[str, bool]:
        chars = []
        for value in ids:
            value = int(value)
            if value == self.EOS:
                return "".join(chars), True
            if value < 3 or value >= len(self):
                raise ValueError(f"Unresolved or invalid output token {value}")
            chars.append(self.tokens[value])
        return "".join(chars), False

    def as_dict(self):
        return {"policy": POLICY, "special_tokens": self.SPECIAL, "characters": self.characters}

    def save(self, path: Path):
        path.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["policy"] != POLICY or data["special_tokens"] != cls.SPECIAL:
            raise ValueError("Incompatible tokenizer policy or special token IDs")
        return cls(data["characters"])
