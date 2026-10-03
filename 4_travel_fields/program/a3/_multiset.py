import sys, numpy as np
sys.path.insert(0,'.')
from a3.harness.a3_gn_precision import trace_precision, trace_production, MODES
from a3.harness.float32_transport_validation import make_medium, gen_ray
ref=MODES["f64ref"]
def norm_multiset(tokens):
    # collapse tie groups: each transaction may carry per-axis plane indices; compare as
    # a sorted multiset of (axis, plane) crossings regardless of transaction grouping/order
    crossings=[]
    for tok in tokens:
        for ax in range(3):
            if tok[ax] is not None:
                crossings.append((ax,tok[ax]))
    return tuple(sorted(crossings))
for F in [16,64,128]:
    med=make_medium(F); rng=np.random.default_rng(500+F)
    order_mism=0; multiset_mism=0; n=0
    for cat in ["random","grazing"]:
        got=0; att=0
        while got<80 and att<4000:
            att+=1
            ro,rd=gen_ray(rng,med,cat)
            if rd is None: continue
            r=trace_precision(med,ro,rd,ref)
            if not r.hit or r.total_tau<=1e-6: continue
            got+=1; n+=1
            p=trace_production(med,ro,rd)
            if p.stream_tokens!=r.stream_tokens: order_mism+=1
            if norm_multiset(p.stream_tokens)!=norm_multiset(r.stream_tokens): multiset_mism+=1
    print("F%3d n=%d  order_mism %d (%.0f%%)  MULTISET_mism %d"%(F,n,order_mism,100*order_mism/n,multiset_mism))
