import sys, numpy as np
sys.path.insert(0,'.')
from a3.harness.a3_gn_precision import trace_precision, MODES, ABLATION
from a3.harness.float32_transport_validation import make_medium, gen_ray
ref=MODES["f64ref"]
CATS=["grazing","near_diagonal","random"]  # tail-bearing mix
errs={k:[] for k in ABLATION}
for F in [64,128]:
    med=make_medium(F); rng=np.random.default_rng(303+F)
    for cat in CATS:
        got=0; att=0
        while got<60 and att<3000:
            att+=1
            ro,rd=gen_ray(rng,med,cat)
            if rd is None: continue
            r=trace_precision(med,ro,rd,ref)
            if not r.hit or r.total_tau<=1e-6: continue
            got+=1
            for k,cfg in ABLATION.items():
                m=trace_precision(med,ro,rd,cfg)
                errs[k].append(abs(m.total_tau-r.total_tau))
print("ablation (vs true f64), %d rays:"%len(errs['full32']))
for k in ["full32","A_ray64","B_box64","C_dda64","D_decide64"]:
    a=np.array(errs[k]); tail=int((a>1.5e-6).sum())
    print("  %-12s max %.2e  p99 %.2e  tail(>1.5e-6) %d"%(k,a.max(),np.percentile(a,99),tail))
