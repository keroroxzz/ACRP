import sys, numpy as np
sys.path.insert(0,'.')
from a3.harness.a3_gn_precision import trace_precision, trace_production, MODES
from a3.harness.float32_transport_validation import make_medium, gen_ray
ref=MODES["f64ref"]

def long_ray(rng, med):
    # through the full diagonal, origin far outside
    d=np.array([1.0,1.0,1.0]); d=d/np.linalg.norm(d)
    d=d+rng.normal(size=3)*0.05; d/=np.linalg.norm(d)
    c=0.5*(med.bmin+med.bmax); ro=c-d*3*np.linalg.norm(med.bmax-med.bmin)
    return ro,d

def box_grazing(rng, med):
    # enter through a face nearly tangentially: small component along entry axis
    ax=rng.integers(3); rd=rng.normal(size=3); rd[ax]=rng.choice([-1,1])*10**rng.uniform(-5,-3)
    rd/=np.linalg.norm(rd)
    ro=rng.uniform(med.bmin,med.bmax); ro[ax]= med.bmin[ax]-0.5 if rd[ax]>0 else med.bmax[ax]+0.5
    return ro,rd

for F in [16,64,128]:
    med=make_medium(F); rng=np.random.default_rng(500+F)
    maxtau=maxevt=0.0; hitm=strm=0; n=0
    gens={"random":lambda: gen_ray(rng,med,"random"),"boxgraze":lambda: box_grazing(rng,med),"long":lambda: long_ray(rng,med)}
    for name,g in gens.items():
        got=0; att=0
        while got<60 and att<3000:
            att+=1
            rr=g(); 
            if rr is None: continue
            ro,rd=rr
            if rd is None: continue
            r=trace_precision(med,ro,rd,ref)
            if not r.hit or r.total_tau<=1e-6: continue
            got+=1; n+=1
            p=trace_production(med,ro,rd)
            if p.hit!=r.hit: hitm+=1
            if p.stream_tokens!=r.stream_tokens: strm+=1
            maxtau=max(maxtau,abs(p.total_tau-r.total_tau))
            tt=0.5*r.total_tau
            re=trace_precision(med,ro,rd,ref,target_tau=tt); pe=trace_production(med,ro,rd,target_tau=tt)
            if re.event_t and pe.event_t: maxevt=max(maxevt,abs(pe.event_t-re.event_t))
    print("F%3d n=%d  max_tau_res %.2e  max_evt_res %.2e  hit_mism %d  stream_mism %d"%(F,n,maxtau,maxevt,hitm,strm))
