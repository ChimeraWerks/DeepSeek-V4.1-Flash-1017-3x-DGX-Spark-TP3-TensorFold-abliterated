"""Stream selected tensors into an independent padded checkpoint, CPU only.

Production inputs are read-only. Output must be a new directory. A complete
receipt is written only after all shards and metadata are durable. This does
not qualify the model numerically or make the upstream engine TP3-compatible.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from math import prod
import os
from pathlib import Path
import struct
from tp3_layout import transformation, extra_name, new_shape, spans, patch_config
from io_budget import Budget, Reader

SIZES = {'BF16':2,'F16':2,'F32':4,'I16':2,'I32':4,'I64':8,'U8':1,'I8':1,'F8_E4M3':1,'F8_E8M0':1}
CHUNK = 8 << 20


def read_header(file):
    nraw=file.read(8)
    if len(nraw)!=8: raise ValueError('Truncated shard')
    n=struct.unpack('<Q',nraw)[0]
    if n>128<<20: raise ValueError('Unexpected header size')
    data=file.read(n)
    if len(data)!=n: raise ValueError('Truncated header')
    return json.loads(data),8+n


def copy_range(src,dst,start,size):
    if start is not None: src.seek(start)
    while size:
        take=min(size,CHUNK)
        data=bytes(take) if start is None else src.read(take)
        if len(data)!=take: raise ValueError('Short tensor read')
        dst.write(data)
        size-=take


def write_shard(source, target, items, *, read_mib_s=0):
    """items: selected original name -> checkpoint-plan tensor descriptor."""
    target=Path(target)
    pending=target.with_suffix('.safetensors.partial')
    if target.exists() or pending.exists(): raise FileExistsError(target)
    records=[]; header={}; offset=0
    with Reader(open(source,'rb'),Budget(read_mib_s)) as src:
        original,base=read_header(src)
        for name,spec in sorted(items.items()):
            actual=original.get(name)
            expected={k:spec[k] for k in ('dtype','shape','data_offsets')}
            if actual!=expected: raise ValueError('Source header changed: '+name)
            op=transformation(name)
            shape=new_shape(spec['shape'],op) if op else spec['shape']
            size=prod(shape)*SIZES[spec['dtype']]
            for output_name in (name,extra_name(name)):
                if output_name is None: continue
                if output_name in header: raise ValueError('Duplicate output: '+output_name)
                header[output_name]={'dtype':spec['dtype'],'shape':shape,'data_offsets':[offset,offset+size]}
                records.append((spec,op,size))
                offset+=size
        encoded=json.dumps(header,separators=(',',':')).encode()
        encoded+=b' '*((-len(encoded))%8)
        with open(pending,'xb') as dst:
            dst.write(struct.pack('<Q',len(encoded))); dst.write(encoded)
            for spec,op,size in records:
                origin=base+spec['data_offsets'][0]
                if op:
                    for at,n in spans(spec['shape'],SIZES[spec['dtype']],op):
                        copy_range(src,dst,None if at is None else origin+at,n)
                else:
                    copy_range(src,dst,origin,size)
            dst.flush(); os.fsync(dst.fileno())
    # Final path never replaces a previously existing artifact.
    os.link(pending,target)
    # Retain .partial hardlink for provenance until explicit cleanup, rather
    # than deleting task resources automatically. Receipt distinguishes it.
    return header,offset


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True)
    p.add_argument('--pollard',required=True)
    p.add_argument('--mia',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--write',action='store_true')
    a=p.parse_args()
    plan_bytes=Path(a.plan).read_bytes(); plan=json.loads(plan_bytes)
    roots={'pollard':Path(a.pollard),'mia':Path(a.mia)}
    groups=defaultdict(dict)
    for name,t in plan['tensors'].items(): groups[(t['source'],t['shard'])][name]=t
    report={'arm':plan['arm'],'input_plan_sha256':hashlib.sha256(plan_bytes).hexdigest(),
            'source_groups':len(groups),'phase':'DRY_RUN','serving_ready':False}
    if not a.write:
        print(json.dumps(report,indent=2)); return
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    receipt=dict(report,phase='MATERIALIZING',shards=[])
    index={}; total=0
    for i,((source,shard),items) in enumerate(sorted(groups.items())):
        name=f'tp3-{i+1:05d}-of-{len(groups):05d}.safetensors'
        hdr,n=write_shard(roots[source]/shard,out/name,items)
        for key in hdr:
            if key in index: raise ValueError('Duplicate final tensor: '+key)
            index[key]=name
        total+=n
        receipt['shards'].append({'file':name,'payload_bytes':n,'tensor_count':len(hdr)})
        (out/'PROGRESS.json').write_text(json.dumps(receipt,indent=2))
        print(f'{i+1}/{len(groups)} {name}',flush=True)
    config=patch_config(json.loads((roots['mia']/'config.json').read_text()))
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                 'generation_config.json','chat_template.jinja','quantization_config.json']:
        src=roots['mia']/name
        if src.exists(): (out/name).write_bytes(src.read_bytes())
    (out/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':index}))
    receipt.update(phase='MATERIALIZED_NOT_QUALIFIED',payload_bytes=total,tensor_count=len(index))
    (out/'MATERIALIZED.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
