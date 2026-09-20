"""CPU numerical tests against independent, full-prefix PyTorch equations.
These tests do NOT establish physical-TPU compatibility or ASR accuracy.
"""
import json, tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file
import jax
import jax.numpy as jnp
from breezesprint25.core import *

torch.set_num_threads(2)
jax.config.update('jax_default_matmul_precision', 'highest')
SPEC = ModelSpec(8,16,4,4,2,3,64,64,16,32,64)


def fixture_weights(spec=SPEC):
    rng=np.random.default_rng(20260920)
    w={}
    def add(k, shape, norm=False):
        w[k] = (np.ones(shape) if norm else rng.normal(0,.1,shape)).astype(np.float32)
    d=spec.width
    for side,n,ff in [('encoder',spec.encoder_layers,spec.encoder_ffn),('decoder',spec.decoder_layers,spec.decoder_ffn)]:
        b='model.'+side
        for i in range(n):
            z=f'{b}.layers.{i}'
            for att in (['self_attn'] if side=='encoder' else ['self_attn','encoder_attn']):
                for proj in ['q_proj','k_proj','v_proj','out_proj']:
                    add(f'{z}.{att}.{proj}.weight',(d,d))
                    if proj!='k_proj': add(f'{z}.{att}.{proj}.bias',(d,))
            for ln in (['self_attn_layer_norm','final_layer_norm'] if side=='encoder' else ['self_attn_layer_norm','final_layer_norm','encoder_attn_layer_norm']):
                add(f'{z}.{ln}.weight',(d,),True); add(f'{z}.{ln}.bias',(d,))
            add(f'{z}.fc1.weight',(ff,d)); add(f'{z}.fc1.bias',(ff,))
            add(f'{z}.fc2.weight',(d,ff)); add(f'{z}.fc2.bias',(d,))
        add(b+'.layer_norm.weight',(d,),True); add(b+'.layer_norm.bias',(d,))
    add('model.encoder.conv1.weight',(d,spec.mel_bins,3));add('model.encoder.conv1.bias',(d,))
    add('model.encoder.conv2.weight',(d,d,3));add('model.encoder.conv2.bias',(d,))
    add('model.encoder.embed_positions.weight',(spec.audio_ctx,d))
    add('model.decoder.embed_positions.weight',(spec.text_ctx,d))
    add('model.decoder.embed_tokens.weight',(spec.vocab,d))
    return w


class Reference:
    def __init__(self, weights, spec=SPEC):
        self.w={k:torch.tensor(v) for k,v in weights.items()}; self.s=spec
    def ln(self,x,b):return F.layer_norm(x,(self.s.width,),self.w[b+'.weight'],self.w[b+'.bias'],1e-5)
    def lin(self,x,b):return F.linear(x,self.w[b+'.weight'],self.w.get(b+'.bias'))
    def attn(self,x,src,b,heads,causal=False):
        n,t,d=x.shape; ss=src.shape[1]; hd=d//heads
        q=self.lin(x,b+'.q_proj').reshape(n,t,heads,hd).transpose(1,2)
        k=self.lin(src,b+'.k_proj').reshape(n,ss,heads,hd).transpose(1,2)
        v=self.lin(src,b+'.v_proj').reshape(n,ss,heads,hd).transpose(1,2)
        scores=(q*hd**-.25) @ (k*hd**-.25).transpose(-1,-2)
        if causal: scores=scores.masked_fill(torch.arange(ss)[None,:]>torch.arange(t)[:,None],float('-inf'))
        out=(scores.softmax(-1) @ v).transpose(1,2).reshape(n,t,d)
        return self.lin(out,b+'.out_proj')
    def enc(self,mel):
        b='model.encoder'
        x=F.gelu(F.conv1d(mel,self.w[b+'.conv1.weight'],self.w[b+'.conv1.bias'],padding=1))
        x=F.gelu(F.conv1d(x,self.w[b+'.conv2.weight'],self.w[b+'.conv2.bias'],padding=1,stride=2)).transpose(1,2)
        x=x+self.w[b+'.embed_positions.weight']
        for i in range(self.s.encoder_layers):
            z=f'{b}.layers.{i}';n=self.ln(x,z+'.self_attn_layer_norm')
            x=x+self.attn(n,n,z+'.self_attn',self.s.encoder_heads)
            x=x+self.lin(F.gelu(self.lin(self.ln(x,z+'.final_layer_norm'),z+'.fc1')),z+'.fc2')
        return self.ln(x,b+'.layer_norm')
    def dec(self,ids,encoded):
        b='model.decoder';x=F.embedding(ids,self.w[b+'.embed_tokens.weight'])+self.w[b+'.embed_positions.weight'][:ids.shape[1]]
        for i in range(self.s.decoder_layers):
            z=f'{b}.layers.{i}';n=self.ln(x,z+'.self_attn_layer_norm')
            x=x+self.attn(n,n,z+'.self_attn',self.s.decoder_heads,True)
            n=self.ln(x,z+'.encoder_attn_layer_norm')
            x=x+self.attn(n,encoded,z+'.encoder_attn',self.s.decoder_heads)
            x=x+self.lin(F.gelu(self.lin(self.ln(x,z+'.final_layer_norm'),z+'.fc1')),z+'.fc2')
        x=self.ln(x,b+'.layer_norm')
        return x @ self.w[b+'.embed_tokens.weight'].T


@pytest.fixture(scope='module')
def shared():
    w=fixture_weights();rng=np.random.default_rng(7)
    mel=rng.normal(0,.3,(2,SPEC.mel_bins,2*SPEC.audio_ctx)).astype(np.float32)
    with tempfile.TemporaryDirectory() as t:
        path=str(Path(t)/'model.safetensors');save_file(w,path)
        p=load_hf_weights(path,SPEC,jax.devices('cpu')[0],dtype='float32')
        ref=Reference(w);enc=ref.enc(torch.tensor(mel))
        yield w,mel,p,ref,enc,path


def test_encoder_matches_independent_torch(shared):
    _,mel,p,_,ref_enc,_=shared
    fn=jax.jit(lambda p,x:encode(p,x,SPEC))
    out=np.asarray(fn(p['enc'],jnp.asarray(mel)))
    np.testing.assert_allclose(out,ref_enc.numpy(),atol=8e-6,rtol=5e-5)


def test_static_cache_matches_full_prefix_at_every_step(shared):
    _,mel,p,ref,enc,_=shared
    e=jnp.asarray(enc.numpy());cross=prepare_cross(p['dec'],e,SPEC)
    cache=empty_cache(SPEC,2,jnp.float32)
    rng=np.random.default_rng(3);tokens=rng.integers(0,40,(2,24),dtype=np.int32)
    fn=jax.jit(lambda p,t,i,c,x:decode_step(p,t,i,c,x,SPEC))
    errors=[]
    for i in range(tokens.shape[1]):
        out,cache=fn(p['dec'],jnp.asarray(tokens[:,i]),jnp.int32(i),cache,cross)
        ref_out=ref.dec(torch.tensor(tokens[:,:i+1],dtype=torch.long),enc)[:,-1,:].numpy()
        errors.append(float(np.max(np.abs(np.asarray(out)-ref_out))))
        np.testing.assert_allclose(np.asarray(out),ref_out,atol=8e-6,rtol=5e-5)
    print('max_full_prefix_logit_error=',max(errors))


def timestamp_reference(logits,tokens,prefix_len,last_ts,sm,bm,max_ts,ds):
    x=np.array(logits,copy=True);x[:,sm]=-np.inf
    if tokens.shape[1]==prefix_len:x[:,bm]=-np.inf
    for row in range(x.shape[0]):
        seq=tokens[row,prefix_len:];last=len(seq)>0 and seq[-1]>=ds.timestamp_begin
        prev=len(seq)<2 or seq[-2]>=ds.timestamp_begin
        x[row,ds.no_timestamps]=-np.inf
        if last:
            if prev:x[row,ds.timestamp_begin:]=-np.inf
            else:x[row,:ds.eos]=-np.inf
        stamps=seq[seq>=ds.timestamp_begin]
        if len(stamps):
            minimum=stamps[-1] if last and not prev else stamps[-1]+1
            x[row,ds.timestamp_begin:minimum]=-np.inf
        x[row,max_ts[row]+1:]=-np.inf
        if len(seq)==0:
            x[row,:ds.timestamp_begin]=-np.inf
            x[row,ds.timestamp_begin+ds.max_initial_timestamp+1:]=-np.inf
        score=torch.tensor(x[row])
        if torch.logsumexp(score[ds.timestamp_begin:],0)>score[:ds.timestamp_begin].max():x[row,:ds.timestamp_begin]=-np.inf
    return x


@pytest.mark.parametrize('seq',[[],[48],[48,3],[48,3,51],[48,3,51,51],[48,3,51,51,4],[48,3,51,55,4,60]])
def test_timestamp_rules_match_independent_reference(seq):
    ds=DecodeSpec(40,44,48,47);rng=np.random.default_rng(123)
    logits=rng.normal(size=(1,64)).astype(np.float32);sm=np.zeros(64,bool);sm[[41,42,43,44,47]]=True
    bm=np.zeros(64,bool);bm[[40,0]]=True
    prefix=[41,42,43];tokens=np.array([prefix+seq],np.int32)
    buf=np.full((1,32),40,np.int32);buf[:,:tokens.shape[1]]=tokens
    stamp=[x for x in seq if x>=48];last=np.array([stamp[-1] if stamp else -1],np.int32)
    max_ts=np.array([63],np.int32)
    actual=np.asarray(jax.jit(lambda x,b,pos,lt:filter_logits(x,b,pos,3,lt,sm,bm,max_ts,ds))(logits,buf,jnp.int32(tokens.shape[1]),last))
    expected=timestamp_reference(logits,tokens,3,last,sm,bm,max_ts,ds)
    np.testing.assert_array_equal(actual,expected)


def test_device_loop_matches_reference_greedy(shared):
    _,_,p,ref,enc,_=shared
    ds=DecodeSpec(40,44,48,47);sm=np.zeros(64,bool);sm[[41,42,43,44,47]]=True
    bm=np.zeros(64,bool);bm[[0,40]]=True
    pr=np.array([[41,42,43],[41,42,43]],np.int32);mt=np.array([63,63],np.int32)
    encj=jnp.asarray(enc.numpy())
    fn=jax.jit(lambda p,x,pr,sm,bm,mt:generate(p,x,pr,sm,bm,mt,SPEC,ds))
    actual=jax.device_get(fn(p['dec'],encj,pr,sm,bm,mt))
    buffer=pr.copy();end=np.zeros(2,bool)
    for i in range(3,SPEC.text_ctx):
        logits=ref.dec(torch.tensor(buffer,dtype=torch.long),enc)[:,-1,:].numpy()
        filt=timestamp_reference(logits,buffer,3,None,sm,bm,mt,ds)
        nxt=np.argmax(filt,axis=-1);nxt[end]=ds.eos;end |= nxt==ds.eos
        buffer=np.column_stack([buffer,nxt])
        if end.all():break
    np.testing.assert_array_equal(actual['sequences'][:,:buffer.shape[1]],buffer)
    np.testing.assert_array_equal(actual['ended'],end)
    repeated=jax.device_get(fn(p['dec'],encj,pr,sm,bm,mt))
    np.testing.assert_array_equal(actual['sequences'],repeated['sequences'])
    assert actual['valid'].all()


def test_missing_weight_is_fatal(shared,tmp_path):
    w=dict(shared[0]);w.pop('model.decoder.layers.0.self_attn.q_proj.weight')
    path=str(tmp_path/'bad.safetensors');save_file(w,path)
    with pytest.raises(ValueError,match='schema mismatch'):load_hf_weights(path,SPEC,jax.devices('cpu')[0],'float32')


def test_wrong_shape_is_fatal(shared,tmp_path):
    w=dict(shared[0]);w['model.encoder.conv1.weight']=w['model.encoder.conv1.weight'][:,:,:2]
    path=str(tmp_path/'bad.safetensors');save_file(w,path)
    with pytest.raises(ValueError,match='Wrong weight shape'):load_hf_weights(path,SPEC,jax.devices('cpu')[0],'float32')


def test_nonfinite_weight_is_fatal(shared,tmp_path):
    w={k:v.copy() for k,v in shared[0].items()};w['model.encoder.conv1.bias'][0]=np.nan
    path=str(tmp_path/'bad.safetensors');save_file(w,path)
    with pytest.raises(ValueError,match='nonfinite'):load_hf_weights(path,SPEC,jax.devices('cpu')[0],'float32')


def test_untied_projection_is_fatal(shared,tmp_path):
    w=dict(shared[0]);w['proj_out.weight']=w['model.decoder.embed_tokens.weight']+1
    path=str(tmp_path/'bad.safetensors');save_file(w,path)
    with pytest.raises(ValueError,match='not tied'):load_hf_weights(path,SPEC,jax.devices('cpu')[0],'float32')


def test_bfloat16_is_finite_and_repeatable(shared):
    _,mel,_,_,_,path=shared
    p=load_hf_weights(path,SPEC,jax.devices('cpu')[0],dtype='bfloat16')
    eng=JaxWhisper(p,SPEC,jax.devices('cpu')[0],'bfloat16');enc=eng.encode(mel)
    ds=DecodeSpec(40,44,48,47);sm=np.zeros(64,bool);sm[[41,42,43,44,47]]=True
    args=(enc,np.array([[41,42,43],[41,42,43]],np.int32),sm,np.zeros(64,bool),np.array([63,63],np.int32),ds)
    a=eng.generate(*args);b=eng.generate(*args)
    assert np.isfinite(np.asarray(enc)).all();assert a['valid'].all()
    np.testing.assert_array_equal(a['sequences'],b['sequences'])
