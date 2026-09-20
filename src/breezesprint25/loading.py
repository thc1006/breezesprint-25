"""Host-packed checkpoint loading; inference equations stay in core.py.

Source tensors are validated, transposed and cast on CPU. Packed layer groups
are transferred once instead of individually cast and stacked on the TPU.
"""
import numpy as np
import ml_dtypes
import jax
from .core import ModelSpec

def load_weights(filename, spec: ModelSpec, device, dtype='bfloat16', index_path=None):
    """Validate every expected tensor before loading, then stream to the device.

    SafeTensors only: no pickle, no trust_remote_code, no random missing weights.
    """
    from safetensors import safe_open
    storage = ml_dtypes.bfloat16 if dtype == 'bfloat16' else np.float32
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
    from .checkpoint import TensorStore
    with TensorStore(filename, index_path=index_path) as sf:
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
            if (a.dtype.kind != 'f' and a.dtype != np.dtype(ml_dtypes.bfloat16)) or not np.isfinite(a).all():
                raise ValueError('Non-floating or nonfinite checkpoint tensor: ' + name)
            if conv: a = a.transpose(2, 1, 0)
            elif transpose: a = a.T
            # Cast on the host: no per-tensor XLA cast executable or synchronization.
            return np.ascontiguousarray(a, dtype=np.float32 if keep_fp32 else storage)

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
            first = layer(side, 0)
            groups = {key: np.empty((count, *value.shape), dtype=value.dtype)
                      for key, value in first.items()}
            for key, value in first.items(): groups[key][0] = value
            del first
            for i in range(1, count):
                member = layer(side, i)
                for key, value in member.items(): groups[key][i] = value
                del member
            stacked = {}
            for key in list(groups):
                group = groups.pop(key)
                stacked[key] = jax.device_put(group, device)
                stacked[key].block_until_ready()
                del group
            p = {'layers': stacked, 'lnw': get(b+'.layer_norm.weight', keep_fp32=True),
                 'lnb': get(b+'.layer_norm.bias', keep_fp32=True),
                 'pos': get(b+'.embed_positions.weight', keep_fp32=True)}
            if side == 'encoder':
                for n in [1, 2]:
                    p[f'c{n}w'] = get(f'{b}.conv{n}.weight', conv=True)
                    p[f'c{n}b'] = get(f'{b}.conv{n}.bias')
            else:
                p['emb'] = get(b+'.embed_tokens.weight')
            result[short] = jax.tree.map(
                lambda x: jax.device_put(x,device) if isinstance(x,np.ndarray) else x, p)
            jax.block_until_ready(result[short])
    return result
