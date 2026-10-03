import sys, numpy as np
sys.path.insert(0,'.')
from a3.harness.a3_gn_precision import trace_precision, MODES
from a3.harness.float32_transport_validation import make_medium, gen_ray, CATEGORIES
ref=MODES["f64ref"]; rq=MODES["mixed64"]
med=make_medium(64); rng=np.random.default_rng(101)
rows=[]
for cat in CATEGORIES:
    got=0; att=0
    while got<70 and att<3000:
        att+=1
        ro,rd=gen_ray(rng,med,cat)
        if rd is None: continue
        r=trace_precision(med,ro,rd,ref)
        if not r.hit or r.total_tau<=1e-6: continue
        got+=1
        q=trace_precision(med,ro,rd,rq)
        g=float(np.min(np.abs(rd/np.linalg.norm(rd))))
        rows.append((g,abs(q.total_tau-r.total_tau)))
rows=np.array(rows)
print("total rays:",len(rows))
for thr in [0,1e-4,3e-4,1e-3,3e-3,1e-2]:
    keep=rows[rows[:,0]>=thr]
    print("  graze>=%.0e : kept %4d (fallback %4.1f%%)  max|dtau| kept = %.2e"%(
        thr,len(keep),100*(1-len(keep)/len(rows)), keep[:,1].max() if len(keep) else 0))
