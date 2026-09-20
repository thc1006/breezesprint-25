"""Decode Whisper byte-BPE tokens without a Transformers/tokenizer.json dependency.

This is deliberately NOT a general-purpose text encoder. Generation needs only
special-token lookup and the literal space token; all ordinary outputs are
decoded directly from the publisher's byte-BPE vocabulary. Multi-byte UTF-8 is
assembled across tokens before decoding, preserving Chinese and mixed English.
"""
from pathlib import Path
from types import SimpleNamespace
import json


def byte_alphabet():
    values = list(range(ord('!'), ord('~')+1)) + list(range(161, 173)) + list(range(174, 256))
    points = values.copy()
    n = 0
    for value in range(256):
        if value not in values:
            values.append(value); points.append(256+n); n += 1
    return {chr(point): value for value, point in zip(values, points)}


class DecoderVocabulary:
    def __init__(self, vocab_path, added_tokens_path=None):
        self.vocab = json.loads(Path(vocab_path).read_text(encoding='utf-8'))
        if not isinstance(self.vocab, dict): raise ValueError('Vocabulary must be a JSON object.')
        if added_tokens_path:
            added = json.loads(Path(added_tokens_path).read_text(encoding='utf-8'))
            if not isinstance(added, dict): raise ValueError('Added tokens must be a JSON object.')
            for token, ident in added.items():
                if token in self.vocab and self.vocab[token] != ident:
                    raise ValueError('Added-token ID conflicts with vocabulary: ' + token)
                self.vocab[token] = ident
        if any(not isinstance(k, str) or type(v) is not int or v < 0 for k,v in self.vocab.items()):
            raise ValueError('Invalid vocabulary token or ID.')
        if len(set(self.vocab.values())) != len(self.vocab):
            raise ValueError('Duplicate vocabulary ID for different tokens.')
        self.inverse = {v:k for k,v in self.vocab.items()}
        self.bytes = byte_alphabet()
        self.eos = self.token_to_id('<|endoftext|>')
        if self.eos is None: raise ValueError('Missing end-of-text token.')
        self.decoded_pieces = {}
        for ident, token in self.inverse.items():
            if ident < self.eos:
                try: self.decoded_pieces[ident] = bytes(self.bytes[c] for c in token)
                except KeyError as exc: raise ValueError('Invalid byte-BPE token.') from exc

    def token_to_id(self, token): return self.vocab.get(token)

    def encode(self, text, add_special_tokens=False):
        if text != ' ' or add_special_tokens:
            raise ValueError('DecoderVocabulary only encodes the literal space without special tokens.')
        space = next(k for k,v in self.bytes.items() if v == 32)
        ident = self.vocab.get(space)
        if ident is None or ident >= self.eos: raise ValueError('Missing ordinary space token.')
        return SimpleNamespace(ids=[ident])

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for item in ids:
            ident = int(item)
            if ident not in self.inverse: raise ValueError('Unknown output token ID.')
            if ident < self.eos: pieces.append(self.decoded_pieces[ident])
            elif not skip_special_tokens: pieces.append(self.inverse[ident].encode('utf-8'))
        return b''.join(pieces).decode('utf-8', errors='replace')


def resolve_no_speech_token(token_to_id, vocab_size):
    """Resolve a no-speech marker from the actual vocabulary, never a guessed ID.

    OpenAI tiktoken calls this <|nospeech|>; the Hugging Face Whisper/Breeze
    vocabularies call the corresponding control token <|nocaptions|>.
    Both spellings are supported. Conflicting aliases are corrupt metadata,
    not an invitation to silently select whichever was checked first.
    """
    names = ('<|nospeech|>', '<|nocaptions|>')
    found = {}
    for name in names:
        ident = token_to_id(name)
        if ident is None:
            continue
        if type(ident) is not int or not 0 <= ident < vocab_size:
            raise ValueError('Invalid no-speech token ID: ' + name)
        found[name] = ident
    if not found:
        raise ValueError('Missing no-speech token: expected <|nospeech|> or <|nocaptions|> in the model vocabulary.')
    if len(set(found.values())) != 1:
        raise ValueError('Conflicting no-speech token aliases: ' + repr(found))
    name = next(iter(found))
    return name, found[name]
