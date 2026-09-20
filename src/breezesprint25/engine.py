"""Thin residency optimization; the model mathematics remain byte-identical."""
import numpy as np
import ml_dtypes
import jax
from .core import JaxWhisper, generate


class ResidentWhisper(JaxWhisper):
    def __init__(self, params, spec, device, codec):
        super().__init__(params, spec, device, 'bfloat16')
        self.codec = codec
        from .assets import profile
        self.language = profile()['language_token']
        self.prefix_host = codec.prefix(self.language)
        codec.suppress.setflags(write=False)
        codec.begin.setflags(write=False)
        self.controls = tuple(jax.device_put(a, device) for a in
                              (self.prefix_host, codec.suppress, codec.begin))
        self.generate_fn = jax.jit(lambda p, x, pr, sm, bm, mt:
                                  generate(p, x, pr, sm, bm, mt, spec, codec.ds))

    def encode(self, features):
        if features.shape != (1, self.spec.mel_bins, 2*self.spec.audio_ctx):
            raise ValueError('Expected a single fixed 30-second feature window.')
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite input features.')
        # Avoid compiling/executing a separate device-cast graph per input shape.
        features = np.ascontiguousarray(features, dtype=ml_dtypes.bfloat16)
        result = self.encoder(self.params['enc'], jax.device_put(features, self.device))
        result.block_until_ready()
        if any(d.platform != self.device.platform for d in result.devices()):
            raise RuntimeError('Encoder output escaped the selected device.')
        return result

    def generate(self, encoded, prefix, suppress_mask, begin_mask, max_timestamp, ds):
        if (not np.array_equal(prefix, self.prefix_host) or
                suppress_mask is not self.codec.suppress or
                begin_mask is not self.codec.begin or ds != self.codec.ds):
            raise ValueError('BreezeSprint25 accepts only its fixed Chinese-output transcription profile.')
        output = self.generate_fn(self.params['dec'], encoded, *self.controls,
                                  jax.device_put(max_timestamp, self.device))
        output = jax.device_get(output)
        if not np.all(output['valid']):
            raise RuntimeError('Invalid decoder logits: no transcript was saved.')
        return output
