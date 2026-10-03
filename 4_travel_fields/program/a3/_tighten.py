import sys, numpy as np, json
sys.path.insert(0,'.')
from a3.harness.a3_gn_precision import trace_precision, MODES
from a3.harness.float32_transport_validation import make_medium, gen_ray, CATEGORIES
ref=MODES["f64ref"]; rq=MODES["mixed64"]
rows=[]  # (m0,m1,m2,dtau)
for F in [64,128]:
    med=make_medium(F); rng=np.random.default_rng(202+F)
    for cat in CATEGORIES:
        got=0; att=0
        while got<45 and att<2500:
            att+=1
            ro,rd=gen_ray(rng,med,cat)
            if rd is None: continue
            r=trace_precision(med,ro,rd,ref)
            if not r.hit or r.total_tau<=1e-6: continue
            got+=1
            q=trace_precision(med,ro,rd,rq)
            m=np.sort(np.abs(rd/np.linalg.norm(rd)))
            rows.append((float(m[0]),float(m[1]),float(m[2]),abs(q.total_tau-r.total_tau)))
rows=np.array(rows); N=len(rows); SAFE=1.5e-6
print("rays",N,"  overall max|dtau| %.2e"%rows[:,3].max())
def evalp(name,flag):
    fl=np.array([flag(r[0],r[1],r[2]) for r in rows],bool)
    kept=rows[~fl]; missed=int((kept[:,3]>SAFE).sum()) if len(kept) else 0
    keptmax=kept[:,3].max() if len(kept) else 0
    print("  %-28s flagged %4.1f%%   kept_max %.2e   MISSED(bad kept) %d"%(
        name,100*fl.mean(),keptmax,missed))
evalp("P1: m0<1e-3", lambda a,b,c: a<1e-3)
evalp("P2: m0<1e-3 & m1>0.1", lambda a,b,c: a<1e-3 and b>0.1)
evalp("P3: m0<1e-3 & m1>0.05", lambda a,b,c: a<1e-3 and b>0.05)
evalp("P4: m0<1e-3 & m1>1e-2", lambda a,b,c: a<1e-3 and b>1e-2)
evalp("P5: m0<3e-3 & m1>0.05", lambda a,b,c: a<3e-3 and b>0.05)
