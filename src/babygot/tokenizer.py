"""A tiny, deterministic tokenizer for child-directed utterances.

It is a whitespace/punctuation word tokenizer with a character-level fallback so
that any unseen word can still be encoded (important for the open vocabulary of
caregiver speech).  No external dependencies; fully pickle-able.

Special tokens:
    <pad> 0, <bos> 1, <eos> 2, <unk> 3, <num> 4 (numbers are canonicalised)
"""

from __future__ import annotations

from typing import Dict, Iterable, List


PAD, BOS, EOS, UNK, NUM = "<pad>", "<bos>", "<eos>", "<unk>", "<num>"
SPECIAL = [PAD, BOS, EOS, UNK, NUM]

_PUNCT = set(".,!?;:'\"()[]")


def _tokenize(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.lower().strip().split():
        tok = raw.strip(".,!?;:'\"()[]")
        if tok == "":
            continue
        if tok.isdigit():
            out.append(NUM)
        else:
            out.append(tok)
    return out


class Tokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab: Dict[str, int] = dict(vocab)
        self.inv: Dict[int, str] = {i: w for w, i in vocab.items()}

    # -- vocabulary construction -------------------------------------------
    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1,
              max_size: int = 1024) -> "Tokenizer":
        counts: Dict[str, int] = {}
        for t in texts:
            for tok in _tokenize(t):
                counts[tok] = counts.get(tok, 0) + 1
        vocab: Dict[str, int] = {w: i for i, w in enumerate(SPECIAL)}
        # keep the most frequent words, then fill remaining slots with a char
        # vocabulary so every word is encodable.
        freq_words = sorted(
            (w for w, c in counts.items() if c >= min_freq),
            key=lambda w: -counts[w],
        )
        for w in freq_words:
            if len(vocab) >= max_size:
                break
            if w not in vocab:
                vocab[w] = len(vocab)
        return cls(vocab)

    # -- encode / decode ----------------------------------------------------
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.vocab[BOS])
        for tok in _tokenize(text):
            ids.append(self.vocab.get(tok, self.vocab[UNK]))
        if add_eos:
            ids.append(self.vocab[EOS])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(
            self.inv.get(int(i), UNK)
            for i in ids
            if int(i) not in (self.vocab[PAD], self.vocab[BOS], self.vocab[EOS])
        )

    def __len__(self) -> int:
        return len(self.vocab)

    def pad_id(self) -> int:
        return self.vocab[PAD]

    def bos_id(self) -> int:
        return self.vocab[BOS]

    def eos_id(self) -> int:
        return self.vocab[EOS]

    def num_id(self) -> int:
        return self.vocab[NUM]

    @property
    def number_tokens(self) -> List[str]:
        """Candidate output strings for counting / 'how many' probes."""
        return [str(i) for i in range(1, 7)] + ["one", "two", "three",
                                                 "four", "five", "six"]
