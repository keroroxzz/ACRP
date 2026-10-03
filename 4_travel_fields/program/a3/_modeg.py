import sys, numpy as np
sys.path.insert(0,'.')
from a3.harness.a3_gn_precision import trace_precision, MODES, ABLATION
from a3.harness.float32_transport_validation import make_medium, gen_ray
ref=MODES["f64ref"]
modes={"full32":ABLATION["full32"],"G_boxonly64":ABLATION["G_boxonly64"],"decision64":MODES["decision64"]}
errs={k:[] for k in modes}
med=make_medium(64); rng=np.random.default_rng(367)
for cat in ["grazing","near_diagonal","random"]:
    got=0; att=0
    while got<45 and att<2500:
        att+=1
        ro,rd=gen_ray(rng,med,cat)
        if rd is None: continue
        r=trace_precision(med,ro,rd,ref)
        if not r.hit or r.total_tau<=1e-6: continue
        got+=1
        for k,cfg in modes.items():
            errs[k].append(abs(trace_precision(med,ro,rd,cfg).total_tau-r.total_tau))
print("F64, %d rays:"%len(errs['full32']))
for k in modes:
    a=np.array(errs[k]); print("  %-14s max %.2e  p99 %.2e  tail(>1.5e-6) %d"%(k,a.max(),np.percentile(a,99),int((a>1.5e-6).sum())))
