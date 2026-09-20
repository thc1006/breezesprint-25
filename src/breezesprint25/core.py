"""Small, inference-only Whisper engine using public JAX primitives.

Original implementation of the Whisper equations. See THIRD_PARTY_NOTICES.md.
No PyTorch, Transformers, Flax, CUDA, custom call, or remote Python model code.
The production entry point MUST select a real TPU; CPU is for unit tests only.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class ModelSpec:
    mel_bins: int
    width: int
    encoder_heads: int
    decoder_heads: int
    encoder_layers: int
    decoder_layers: int
    encoder_ffn: int
    decoder_ffn: int
    audio_ctx: int
    text_ctx: int
    vocab: int

    @classmethod
    def from_config(cls, c: dict) -> 'ModelSpec':
        if c.get('model_type') != 'whisper':
            raise ValueError('Only standard Whisper checkpoints are accepted.')
        if c.get('activation_function', 'gelu') != 'gelu' or c.get('scale_embedding', False):
            raise ValueError('Nonstandard activation/embedding scaling is not supported.')
        if not c.get('tie_word_embeddings', True):
            raise ValueError('Untied output projection is not supported.')
        spec = cls(int(c['num_mel_bins']), int(c['d_model']),
                   int(c['encoder_attention_heads']), int(c['decoder_attention_heads']),
                   int(c['encoder_layers']), int(c['decoder_layers']),
                   int(c['encoder_ffn_dim']), int(c['decoder_ffn_dim']),
                   int(c['max_source_positions']), int(c['max_target_positions']),
                   int(c['vocab_size']))
        if min(spec.__dict__.values()) <= 0:
            raise ValueError('Invalid nonpositive model dimensions.')
        if spec.width % spec.encoder_heads or spec.width % spec.decoder_heads:
            raise ValueError('Attention head dimensions must divide model width.')
        return spec


def norm(x, weight, bias):
    """FP32 layer-normalization statistics, including in BF16 mode."""
    z = x.astype(jnp.float32)
    mu = jnp.mean(z, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(z - mu), axis=-1, keepdims=True)
    return ((z - mu) * lax.rsqrt(var + 1e-5) * weight + bias).astype(x.dtype)


def linear(x, w, b=None):
    out = jnp.matmul(x, w)
    return out if b is None else out + b.astype(out.dtype)


def attention(q, k, v, allowed=None):
    """q/k/v: [batch, time, heads, head_dim]; softmax is FP32."""
    scale = q.shape[-1] ** -0.25
    scores = jnp.einsum('bthd,bshd->bhts', q * scale, k * scale,
                        preferred_element_type=jnp.float32)
    if allowed is not None:
        scores = jnp.where(allowed, scores, -jnp.inf)
    probs = jax.nn.softmax(scores, axis=-1).astype(q.dtype)
    out = jnp.einsum('bhts,bshd->bthd', probs, v)
    return out.reshape(q.shape[0], q.shape[1], -1)


def feed_forward(x, p):
    h = norm(x, p['ln2w'], p['ln2b'])
    h = jax.nn.gelu(linear(h, p['f1'], p['b1']), approximate=False)
    return x + linear(h, p['f2'], p['b2'])


def encode(p, mel, spec: ModelSpec):
    # mel [batch, mel_bins, 2*audio_ctx]. Explicit padding is essential:
    # stride-2 SAME padding is NOT equivalent to PyTorch padding=1 here.
    x = jnp.swapaxes(mel, 1, 2)
    x = lax.conv_general_dilated(x, p['c1w'], (1,), ((1, 1),),
                                dimension_numbers=('NWC', 'WIO', 'NWC'))
    x = jax.nn.gelu(x + p['c1b'], approximate=False)
    x = lax.conv_general_dilated(x, p['c2w'], (2,), ((1, 1),),
                                dimension_numbers=('NWC', 'WIO', 'NWC'))
    x = jax.nn.gelu(x + p['c2b'], approximate=False)
    x = (x + p['pos']).astype(x.dtype)
    hdim = spec.width // spec.encoder_heads

    def block(x, layer):
        n = norm(x, layer['ln1w'], layer['ln1b'])
        shape = (x.shape[0], x.shape[1], spec.encoder_heads, hdim)
        q = linear(n, layer['q'], layer['qb']).reshape(shape)
        k = linear(n, layer['k']).reshape(shape)
        v = linear(n, layer['v'], layer['vb']).reshape(shape)
        x = x + linear(attention(q, k, v), layer['o'], layer['ob'])
        return feed_forward(x, layer), None

    x, _ = lax.scan(block, x, p['layers'])
    return norm(x, p['lnw'], p['lnb'])


def prepare_cross(p, encoded, spec: ModelSpec):
    shape = (encoded.shape[0], encoded.shape[1], spec.decoder_heads,
             spec.width // spec.decoder_heads)

    def one(_, layer):
        k = linear(encoded, layer['ck']).reshape(shape)
        v = linear(encoded, layer['cv'], layer['cvb']).reshape(shape)
        return None, (k, v)

    _, cross = lax.scan(one, None, p['layers'])
    return cross


def empty_cache(spec: ModelSpec, batch: int, dtype):
    shape = (spec.decoder_layers, batch, spec.text_ctx,
             spec.decoder_heads, spec.width // spec.decoder_heads)
    return jnp.zeros(shape, dtype), jnp.zeros(shape, dtype)


def decode_step(p, tokens, position, cache, cross, spec: ModelSpec):
    """A fixed-shape, one-token decode step. No growing concatenations."""
    x = (p['emb'][tokens] + p['pos'][position]).astype(p['emb'].dtype)[:, None, :]
    shape = (tokens.shape[0], 1, spec.decoder_heads, spec.width // spec.decoder_heads)
    allowed = (jnp.arange(spec.text_ctx) <= position)[None, None, None, :]

    def block(x, values):
        layer, old_k, old_v, cross_k, cross_v = values
        n = norm(x, layer['ln1w'], layer['ln1b'])
        q = linear(n, layer['q'], layer['qb']).reshape(shape)
        k = linear(n, layer['k']).reshape(shape)
        v = linear(n, layer['v'], layer['vb']).reshape(shape)
        new_k = lax.dynamic_update_slice(old_k, k, (0, position, 0, 0))
        new_v = lax.dynamic_update_slice(old_v, v, (0, position, 0, 0))
        x = x + linear(attention(q, new_k, new_v, allowed), layer['o'], layer['ob'])
        n = norm(x, layer['clnw'], layer['clnb'])
        q = linear(n, layer['cq'], layer['cqb']).reshape(shape)
        x = x + linear(attention(q, cross_k, cross_v), layer['co'], layer['cob'])
        return feed_forward(x, layer), (new_k, new_v)

    x, new_cache = lax.scan(block, x, (p['layers'], cache[0], cache[1], cross[0], cross[1]))
    x = norm(x[:, 0, :], p['lnw'], p['lnb'])
    logits = jnp.matmul(x, p['emb'].T, preferred_element_type=jnp.float32)
    return logits, new_cache


@dataclass(frozen=True)
class DecodeSpec:
    eos: int
    no_speech: int
    timestamp_begin: int
    no_timestamps: int
    timestamps: bool = True
    max_initial_timestamp: int = 50  # 1 second / 0.02


def filter_logits(logits, buffer, position, prefix_len, last_timestamp,
                  suppress_mask, begin_mask, max_timestamp, ds: DecodeSpec):
    """JIT-compatible timestamp pairing, monotonicity, and mass rules.

    Follows the rules described in OpenAI Whisper's ApplyTimestampRules.
    Unlike a Python .tolist() loop, this operates entirely on the device.
    """
    ids = jnp.arange(logits.shape[-1])[None, :]
    logits = jnp.where(suppress_mask[None, :], -jnp.inf, logits)
    first = position == prefix_len
    logits = jnp.where(first & begin_mask[None, :], -jnp.inf, logits)
    if not ds.timestamps:
        return jnp.where(ids >= ds.timestamp_begin, -jnp.inf, logits)
    logits = jnp.where(ids == ds.no_timestamps, -jnp.inf, logits)
    last = buffer[:, position - 1]
    prev = buffer[:, jnp.maximum(position - 2, 0)]
    last_was = (position > prefix_len) & (last >= ds.timestamp_begin)
    penultimate_was = (position - prefix_len < 2) | (prev >= ds.timestamp_begin)
    # Following two timestamp tokens, text (or EOS) must follow.
    logits = jnp.where((last_was & penultimate_was)[:, None] &
                       (ids >= ds.timestamp_begin), -jnp.inf, logits)
    # After the closing timestamp of a text segment, require EOS or timestamp.
    logits = jnp.where((last_was & ~penultimate_was)[:, None] &
                       (ids < ds.eos), -jnp.inf, logits)
    minimum = last_timestamp + jnp.where(last_was & ~penultimate_was, 0, 1)
    logits = jnp.where((last_timestamp >= 0)[:, None] &
                       (ids >= ds.timestamp_begin) & (ids < minimum[:, None]),
                       -jnp.inf, logits)
    logits = jnp.where(ids > max_timestamp[:, None], -jnp.inf, logits)
    logits = jnp.where(first & (ids < ds.timestamp_begin), -jnp.inf, logits)
    logits = jnp.where(first & (ids > ds.timestamp_begin + ds.max_initial_timestamp),
                       -jnp.inf, logits)
    # Normalization cancels in this comparison; use logits to avoid NaNs.
    ts_mass = jax.scipy.special.logsumexp(logits[:, ds.timestamp_begin:], axis=-1)
    best_text = jnp.max(logits[:, :ds.timestamp_begin], axis=-1)
    logits = jnp.where((ts_mass > best_text)[:, None] & (ids < ds.timestamp_begin),
                       -jnp.inf, logits)
    return logits


def generate(p, encoded, prefix, suppress_mask, begin_mask, max_timestamp,
             spec: ModelSpec, ds: DecodeSpec):
    """Static KV cache + device-side autoregressive loop. Greedy, no sampling."""
    batch, prefix_len = prefix.shape
    cross = prepare_cross(p, encoded, spec)
    cache = empty_cache(spec, batch, p['emb'].dtype)
    logits = jnp.zeros((batch, spec.vocab), jnp.float32)
    no_speech = jnp.zeros((batch,), jnp.float32)

    def prefill(i, state):
        logits, cache, no_speech = state
        logits, cache = decode_step(p, prefix[:, i], i, cache, cross, spec)
        no_speech = jnp.where(i == 0, jax.nn.softmax(logits, axis=-1)[:, ds.no_speech], no_speech)
        return logits, cache, no_speech

    logits, cache, no_speech = lax.fori_loop(0, prefix_len, prefill, (logits, cache, no_speech))
    buffer = jnp.full((batch, spec.text_ctx), ds.eos, jnp.int32)
    buffer = buffer.at[:, :prefix_len].set(prefix)
    # position, token buffer, KV, logits, ended, sum logprob, lengths, last ts, finite
    state = (jnp.int32(prefix_len), buffer, cache, logits, jnp.zeros(batch, bool),
             jnp.zeros(batch, jnp.float32), jnp.zeros(batch, jnp.int32),
             jnp.full((batch,), -1, jnp.int32), jnp.ones(batch, bool))

    def cond(s):
        return (s[0] < spec.text_ctx) & ~jnp.all(s[4])

    def body(s):
        pos, buf, kv, logits, ended, score, lengths, last_ts, valid = s
        filtered = filter_logits(logits, buf, pos, prefix_len, last_ts,
                                 suppress_mask, begin_mask, max_timestamp, ds)
        row_ok = jnp.any(jnp.isfinite(filtered), axis=-1)
        safe = jnp.where(row_ok[:, None], filtered, jnp.zeros_like(filtered))
        chosen = jnp.argmax(safe, axis=-1).astype(jnp.int32)
        chosen = jnp.where(ended | ~row_ok, ds.eos, chosen)
        lp = jnp.take_along_axis(jax.nn.log_softmax(safe), chosen[:, None], axis=-1)[:, 0]
        active = ~ended
        score = score + jnp.where(active, lp, 0)
        lengths = lengths + active.astype(jnp.int32)
        valid = valid & (~active | row_ok) & (~active | jnp.isfinite(lp))
        buf = buf.at[:, pos].set(chosen)
        last_ts = jnp.where(active & (chosen >= ds.timestamp_begin), chosen, last_ts)
        ended = ended | (chosen == ds.eos)
        # Do not write a KV entry beyond the model context limit.
        logits, kv = lax.cond((pos + 1 < spec.text_ctx) & ~jnp.all(ended),
                              lambda _: decode_step(p, chosen, pos, kv, cross, spec),
                              lambda _: (logits, kv), operand=None)
        return pos + 1, buf, kv, logits, ended, score, lengths, last_ts, valid

    end = lax.while_loop(cond, body, state)
    return {'sequences': end[1], 'lengths': end[6],
            'avg_logprob': end[5] / jnp.maximum(end[6], 1),
            'no_speech_prob': no_speech, 'ended': end[4], 'valid': end[8],
            'truncated': ~end[4], 'prefix_len': jnp.int32(prefix_len)}


class JaxWhisper:
    def __init__(self, params: dict, spec: ModelSpec, device, dtype='bfloat16'):
        self.params, self.spec, self.device = params, spec, device
        self.dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32
        self.encoder = jax.jit(lambda p, x: encode(p, x, spec))
        self._generators: dict[DecodeSpec, Any] = {}

    def encode(self, features: np.ndarray):
        expected = (self.spec.mel_bins, 2 * self.spec.audio_ctx)
        if features.ndim != 3 or features.shape[1:] != expected:
            raise ValueError(f'Expected [batch,{expected[0]},{expected[1]}], got {features.shape}')
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite audio features.')
        x = jax.device_put(features, self.device).astype(self.dtype)
        out = self.encoder(self.params['enc'], x)
        out.block_until_ready()
        if any(d.platform != self.device.platform for d in out.devices()):
            raise RuntimeError('Encoder output is not on the requested backend.')
        return out

    def generate(self, encoded, prefix, suppress_mask, begin_mask, max_timestamp, ds):
        if ds not in self._generators:
            self._generators[ds] = jax.jit(lambda p, x, pr, sm, bm, mt:
                generate(p, x, pr, sm, bm, mt, self.spec, ds))
        args = [jax.device_put(a, self.device) for a in
                (prefix, suppress_mask, begin_mask, max_timestamp)]
        outputs = self._generators[ds](self.params['dec'], encoded, *args)
        outputs = jax.device_get(outputs)  # synchronization included in timing
        if not np.all(outputs['valid']):
            raise RuntimeError('Nonfinite/all-suppressed decoder logits: refusing to save a transcript.')
        return outputs


def load_hf_weights(filename: str, spec: ModelSpec, device, dtype='bfloat16'):
    """Validate every expected tensor before loading, then stream to the device.

    SafeTensors only: no pickle, no trust_remote_code, no random missing weights.
    """
    from safetensors import safe_open
    storage = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32
    d = spec.width
    expected = {}
    def add(name, shape):
        expected[name] = tuple(shape)
    for side, layers, ffn in [('encoder', spec.encoder_layers, spec.encoder_ffn),
                              ('decoder', spec.decoder_layers, spec.decoder_ffn)]:
        base = 'model.' + side
        for i in range(layers):
            b = f'{base}.layers.{i}'
            for att in (['self_attn'] if side == 'encoder' else ['self_attn', 'encoder_attn']):
                for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    add(f'{b}.{att}.{proj}.weight', (d, d))
                    if proj != 'k_proj':
                        add(f'{b}.{att}.{proj}.bias', (d,))
            norms = ['self_attn_layer_norm', 'final_layer_norm']
            if side == 'decoder': norms += ['encoder_attn_layer_norm']
            for ln in norms:
                add(f'{b}.{ln}.weight', (d,)); add(f'{b}.{ln}.bias', (d,))
            add(f'{b}.fc1.weight', (ffn, d)); add(f'{b}.fc1.bias', (ffn,))
            add(f'{b}.fc2.weight', (d, ffn)); add(f'{b}.fc2.bias', (d,))
        add(base + '.layer_norm.weight', (d,)); add(base + '.layer_norm.bias', (d,))
    add('model.encoder.conv1.weight', (d, spec.mel_bins, 3))
    add('model.encoder.conv1.bias', (d,))
    add('model.encoder.conv2.weight', (d, d, 3))
    add('model.encoder.conv2.bias', (d,))
    add('model.encoder.embed_positions.weight', (spec.audio_ctx, d))
    add('model.decoder.embed_positions.weight', (spec.text_ctx, d))
    add('model.decoder.embed_tokens.weight', (spec.vocab, d))
    with safe_open(filename, framework='np') as sf:
        keys = set(sf.keys())
        missing = set(expected) - keys
        extra = keys - set(expected) - {'proj_out.weight'}
        if missing or extra:
            raise ValueError(f'Weight schema mismatch. Missing={sorted(missing)} Extra={sorted(extra)}')
        for name, shape in expected.items():
            if tuple(sf.get_slice(name).get_shape()) != shape:
                raise ValueError(f'Wrong weight shape: {name}: expected {shape}')
        if 'proj_out.weight' in keys:
            if not np.array_equal(sf.get_tensor('proj_out.weight'), sf.get_tensor('model.decoder.embed_tokens.weight')):
                raise ValueError('Output projection is not tied to token embeddings.')

        def get(name, transpose=False, keep_fp32=False, conv=False):
            a = sf.get_tensor(name)
            if a.dtype.kind != 'f' or not np.isfinite(a).all():
                raise ValueError('Non-floating or nonfinite checkpoint tensor: ' + name)
            if conv: a = a.transpose(2, 1, 0)
            elif transpose: a = a.T
            out = jax.device_put(np.ascontiguousarray(a), device)
            out = out.astype(jnp.float32 if keep_fp32 else storage)
            out.block_until_ready()
            return out

        def layer(side, i):
            b = f'model.{side}.layers.{i}'
            p = {}
            for out, origin in [('q','q_proj'), ('k','k_proj'), ('v','v_proj'), ('o','out_proj')]:
                p[out] = get(f'{b}.self_attn.{origin}.weight', transpose=True)
                if out != 'k': p[out+'b'] = get(f'{b}.self_attn.{origin}.bias')
            for out, origin in [('ln1','self_attn_layer_norm'), ('ln2','final_layer_norm')]:
                p[out+'w'] = get(f'{b}.{origin}.weight', keep_fp32=True)
                p[out+'b'] = get(f'{b}.{origin}.bias', keep_fp32=True)
            for n in [1, 2]:
                p[f'f{n}'] = get(f'{b}.fc{n}.weight', transpose=True)
                p[f'b{n}'] = get(f'{b}.fc{n}.bias')
            if side == 'decoder':
                for out, origin in [('cq','q_proj'), ('ck','k_proj'), ('cv','v_proj'), ('co','out_proj')]:
                    p[out] = get(f'{b}.encoder_attn.{origin}.weight', transpose=True)
                    if out != 'ck': p[out+'b'] = get(f'{b}.encoder_attn.{origin}.bias')
                p['clnw'] = get(f'{b}.encoder_attn_layer_norm.weight', keep_fp32=True)
                p['clnb'] = get(f'{b}.encoder_attn_layer_norm.bias', keep_fp32=True)
            return p

        result = {}
        for side, short, count in [('encoder','enc',spec.encoder_layers), ('decoder','dec',spec.decoder_layers)]:
            b = 'model.' + side
            # Stacking one group at a time limits transient HBM. No random init.
            members = [layer(side, i) for i in range(count)]
            stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *members)
            jax.block_until_ready(stacked)
            del members
            p = {'layers': stacked, 'lnw': get(b+'.layer_norm.weight', keep_fp32=True),
                 'lnb': get(b+'.layer_norm.bias', keep_fp32=True),
                 'pos': get(b+'.embed_positions.weight', keep_fp32=True)}
            if side == 'encoder':
                for n in [1, 2]:
                    p[f'c{n}w'] = get(f'{b}.conv{n}.weight', conv=True)
                    p[f'c{n}b'] = get(f'{b}.conv{n}.bias')
            else:
                p['emb'] = get(b+'.embed_tokens.weight')
            result[short] = p
    return result
