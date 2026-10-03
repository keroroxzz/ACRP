
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import numpy as np

def fade_poly(a,b):
    return np.array([3*a*a-2*a*a*a, 6*a*b*(1-a), 3*b*b*(1-2*a), -2*b*b*b], float)

def pad(p,n):
    out=np.zeros(n); out[:len(p)]=p; return out

def build_density_poly(local_entry,direction,corners):
    fx,fy,fz=[fade_poly(local_entry[i],direction[i]) for i in range(3)]
    fxy=np.polynomial.polynomial.polymul(fx,fy)
    fxz=np.polynomial.polynomial.polymul(fx,fz)
    fyz=np.polynomial.polynomial.polymul(fy,fz)
    fxyz=np.polynomial.polynomial.polymul(fxy,fz)
    d000,d100,d010,d110,d001,d101,d011,d111 = (
        corners[0,0,0],corners[1,0,0],corners[0,1,0],corners[1,1,0],
        corners[0,0,1],corners[1,0,1],corners[0,1,1],corners[1,1,1]
    )
    a=d000
    ax=d100-d000; ay=d010-d000; az=d001-d000
    axy=d110-d100-d010+d000
    axz=d101-d100-d001+d000
    ayz=d011-d010-d001+d000
    axyz=d111-d110-d101-d011+d100+d010+d001-d000
    rho=np.zeros(10); rho[0]=a
    rho += ax*pad(fx,10)+ay*pad(fy,10)+az*pad(fz,10)
    rho += axy*pad(fxy,10)+axz*pad(fxz,10)+ayz*pad(fyz,10)
    rho += axyz*pad(fxyz,10)
    return rho

def shift_poly(c,delta):
    out=np.zeros_like(c)
    for k in range(len(c)):
        for j in range(k+1):
            out[j]+=c[k]*math.comb(k,j)*(delta**(k-j))
    return out

def integrate_poly(c,L,sigma):
    acc=c[-1]/len(c)
    for k in range(len(c)-2,-1,-1):
        acc=acc*L+c[k]/(k+1)
    return sigma*L*acc

@dataclass
class Octave:
    lattice: np.ndarray
    frequency: int
    amplitude: float
    @property
    def cells(self): return np.array(self.lattice.shape)-1
    def cell_local(self,p):
        q=p*self.frequency
        cell=np.floor(q).astype(int)
        for a in range(3):
            if cell[a]==self.cells[a]: cell[a]-=1
        return cell, np.clip(q-cell,0,1)
    def corners(self,cell):
        x,y,z=map(int,cell)
        return self.amplitude*self.lattice[x:x+2,y:y+2,z:z+2]

@dataclass
class Medium:
    octaves: List[Octave]
    sigma_t: float
    world_cells: np.ndarray
    @property
    def bmin(self): return np.zeros(3)
    @property
    def bmax(self): return self.world_cells.astype(float)

def intersect_box(ro,rd,bmin,bmax):
    te,tx=-math.inf,math.inf
    for a in range(3):
        if abs(rd[a])<1e-15:
            if ro[a]<bmin[a] or ro[a]>bmax[a]: return False,0,0
            continue
        t0=(bmin[a]-ro[a])/rd[a]; t1=(bmax[a]-ro[a])/rd[a]
        te=max(te,min(t0,t1)); tx=min(tx,max(t0,t1))
    te=max(te,0.0)
    return tx>=te,te,tx

def boundaries(medium,level,ro,rd,te,tx):
    f=medium.octaves[level].frequency
    out=[]
    for a in range(3):
        if abs(rd[a])<1e-15: continue
        for k in range(1,int(medium.world_cells[a]*f)):
            t=(k/f-ro[a])/rd[a]
            if te+1e-13<t<tx-1e-13: out.append((float(t),level))
    return out

def merged_events(medium,ro,rd,te,tx):
    raw=[]
    for l in range(len(medium.octaves)): raw += boundaries(medium,l,ro,rd,te,tx)
    raw.sort()
    events=[(te,tuple())]
    i=0
    while i<len(raw):
        t=raw[i][0]; levels=set(); j=i
        while j<len(raw) and abs(raw[j][0]-t)<=2e-12*max(1,abs(t)):
            levels.add(raw[j][1]); j+=1
        events.append((t,tuple(sorted(levels)))); i=j
    events.append((tx,tuple()))
    return events

def robust_cell_local(o,ro,rd,t):
    q=(ro+rd*t)*o.frequency
    # At an integer grid plane, choose the cell in the forward ray direction.
    nearest=np.rint(q)
    close=np.abs(q-nearest)<=2e-11*np.maximum(1.0,np.abs(q))
    q=np.where(close, q + np.sign(rd)*1e-10, q)
    cell=np.floor(q).astype(int)
    cell=np.minimum(np.maximum(cell,0),o.cells-1)
    local=np.clip(q-cell,0.0,1.0)
    return cell,local

def build_octave_poly(o,ro,rd,t):
    cell,local=robust_cell_local(o,ro,rd,t)
    return build_density_poly(local,rd*o.frequency,o.corners(cell))

@dataclass
class Cost:
    segments:int=0
    rebuilds:int=0
    shifts:int=0

def A0(medium,ro,rd,te,tx):
    events=merged_events(medium,ro,rd,te,tx)
    tau=0.0; cost=Cost()
    for i in range(len(events)-1):
        a,b=events[i][0],events[i+1][0]
        if b<=a: continue
        total=np.zeros(10)
        for o in medium.octaves:
            total += build_octave_poly(o,ro,rd,a); cost.rebuilds+=1
        tau += integrate_poly(total,b-a,medium.sigma_t); cost.segments+=1
    return tau,cost

def A1(medium,ro,rd,te,tx):
    events=merged_events(medium,ro,rd,te,tx)
    polys=[build_octave_poly(o,ro,rd,te) for o in medium.octaves]
    cost=Cost(rebuilds=len(polys)); tau=0.0
    for i in range(len(events)-1):
        a,b=events[i][0],events[i+1][0]
        if b<=a: continue
        tau += integrate_poly(np.sum(polys,axis=0),b-a,medium.sigma_t); cost.segments+=1
        if i+1>=len(events)-1: break
        delta=b-a; crossing=set(events[i+1][1])
        for l,o in enumerate(medium.octaves):
            if l in crossing:
                polys[l]=build_octave_poly(o,ro,rd,b); cost.rebuilds+=1
            else:
                polys[l]=shift_poly(polys[l],delta); cost.shifts+=1
    return tau,cost

def random_ray(medium,rng):
    center=.5*(medium.bmin+medium.bmax)
    if rng.random()<.35: ro=rng.uniform(medium.bmin,medium.bmax)
    else:
        d=rng.normal(size=3); d/=np.linalg.norm(d)
        ro=center+d*1.5*np.linalg.norm(medium.bmax-medium.bmin)
    target=rng.uniform(medium.bmin,medium.bmax)
    rd=target-ro; rd/=np.linalg.norm(rd)
    return ro,rd

@dataclass
class Summary:
    seed:int
    rays:int
    octaves:int
    frequencies:list
    max_tau_error:float
    p99_tau_error:float
    A0_mean_segments:float
    A0_mean_rebuilds:float
    A1_mean_segments:float
    A1_mean_rebuilds:float
    A1_mean_shifts:float
    rebuild_reduction:float
    eq_reduction_shift025:float
    eq_reduction_shift050:float
    all_tests_passed:bool

def run(seed=20260718,rays=300):
    rng=np.random.default_rng(seed)
    world=np.array([3,3,3])
    freqs=[1,2,4,8,16]
    amps=[1,.5,.25,.125,.0625]
    octaves=[]
    for f,a in zip(freqs,amps):
        lattice=rng.uniform(0,1,size=tuple(world*f+1))
        lattice[rng.random(size=lattice.shape)<.35]=0
        octaves.append(Octave(lattice,f,a))
    medium=Medium(octaves,1.2,world)

    errs=[]; a0s=[]; a0r=[]; a1s=[]; a1r=[]; a1sh=[]
    for _ in range(rays):
        ro,rd=random_ray(medium,rng)
        hit,te,tx=intersect_box(ro,rd,medium.bmin,medium.bmax)
        if not hit or tx<=te: continue
        t0,c0=A0(medium,ro,rd,te,tx)
        t1,c1=A1(medium,ro,rd,te,tx)
        errs.append(abs(t1-t0))
        a0s.append(c0.segments); a0r.append(c0.rebuilds)
        a1s.append(c1.segments); a1r.append(c1.rebuilds); a1sh.append(c1.shifts)

    m0=float(np.mean(a0r)); m1=float(np.mean(a1r)); msh=float(np.mean(a1sh))
    return Summary(
        seed,len(errs),len(octaves),freqs,
        float(np.max(errs)),float(np.percentile(errs,99)),
        float(np.mean(a0s)),m0,float(np.mean(a1s)),m1,msh,
        1-m1/m0,
        1-(m1+.25*msh)/m0,
        1-(m1+.50*msh)/m0,
        float(np.max(errs))<2e-10 and abs(np.mean(a0s)-np.mean(a1s))<1e-12
    )

def make_report(s):
    return f"""# A1 Incremental Exact Multiscale Traversal

**Status:** {"PASS" if s.all_tests_passed else "NEEDS REVIEW"}

- Rays: `{s.rays}`
- Octaves: `{s.octaves}`
- Frequencies: `{s.frequencies}`

## Exactness

- P99 optical-depth difference: `{s.p99_tau_error:.6e}`
- Maximum optical-depth difference: `{s.max_tau_error:.6e}`

## A0 naive exact common refinement

- Mean segments: `{s.A0_mean_segments:.3f}`
- Mean octave polynomial rebuilds: `{s.A0_mean_rebuilds:.3f}`

## A1 incremental exact common refinement

- Mean segments: `{s.A1_mean_segments:.3f}`
- Mean octave polynomial rebuilds: `{s.A1_mean_rebuilds:.3f}`
- Mean degree-9 polynomial shifts: `{s.A1_mean_shifts:.3f}`
- Raw rebuild reduction: `{100*s.rebuild_reduction:.2f}%`

If one polynomial shift costs 0.25 of a rebuild, estimated equivalent reduction:
`{100*s.eq_reduction_shift025:.2f}%`.

If one polynomial shift costs 0.50 of a rebuild, estimated equivalent reduction:
`{100*s.eq_reduction_shift050:.2f}%`.

A1 preserves the exact common-refinement partition and exact degree-10
optical depth. Only octaves crossing a boundary are rebuilt; the rest are
reparameterized to the next segment origin.
"""

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=20260718)
    p.add_argument("--rays",type=int,default=300)
    p.add_argument("--json",type=Path)
    p.add_argument("--report",type=Path)
    a=p.parse_args(); s=run(a.seed,a.rays)
    print(json.dumps(asdict(s),indent=2))
    if a.json: a.json.write_text(json.dumps(asdict(s),indent=2),encoding="utf-8")
    if a.report: a.report.write_text(make_report(s),encoding="utf-8")
