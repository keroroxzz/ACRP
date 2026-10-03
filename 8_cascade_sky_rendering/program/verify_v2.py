"""V2 GPU parity, zero/full queues, odd dimensions and history independence."""
import hashlib
import json
import shutil
import numpy as np
from OpenGL.GL import *
from cascade import HERE, display_color
from v2 import V2Renderer


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'source').mkdir(exist_ok=True)
    files=('v2.py','v2_sources.py','single_frame_v2.py','benchmark.py','verify_v2.py','single_frame.py','optimized_sources.py',
           'cascade.py','baseline_psms.glsl','parent_score.frag','scored_reconstruct.frag',
           'cubic_reconstruct.frag','reconstruct.frag','stencil.frag')
    result={'hashes':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in files},'cases':[]}
    for f in files:
        shutil.copyfile(HERE/f,out/'source'/f)
    r=V2Renderer(801,451)
    cases=[dict(camera=(25000,0,0),bias=.4,sun=(0,.9998477,.0174524),itime=10),
           dict(camera=(-43000,17500,1200),bias=-.4,sun=(.6,.7998,.01745),itime=37.9),
           dict(camera=(250000,-330000,2700),bias=-.5,sun=(0,.9998477,-.0174524),itime=83.3)]
    variant=dict(optimized=True,levels=2,factor=4,threshold=.035)
    result.update(gpu=r.gpu,resolution=[r.w,r.h])
    try:
        # Exercise V2 before any reference draw or benchmark warmup.
        r.scene(**cases[0]); cold,_=r.capture(**variant,kernel='opaque')
        warm,_=r.capture(**variant,kernel='opaque')
        assert np.array_equal(cold,warm),'Cold image differs'
        result['cold_warm_bit_identical']=True
        for scene in cases:
            r.scene(**scene)
            ref,_=r.capture(**variant)
            row={'scene':scene,'variants':{}}
            for kernel in ('bounds','opaque','combined','lazy','lazy-bounds'):
                img,_=r.capture(**variant,kernel=kernel)
                rmse=float(np.sqrt(np.mean((display_color(img)-display_color(ref))**2)))
                maximum=float(np.abs(img-ref).max())
                row['variants'][kernel]=dict(max_hdr=maximum,display_rmse=rmse,bit_identical=bool(np.array_equal(img,ref)))
                assert rmse<1e-5,(kernel,rmse)
            result['cases'].append(row)
        r.scene(**cases[0])
        queue_checks=[]
        for force in (-1,0,1):
            ref,work=r.capture(**variant,force=force)
            for mode in (0,32,64,128):
                img,_=r.capture(**variant,queue=mode,force=force)
                err=float(np.abs(img-ref).max())
                assert err<1e-5,('queue output',mode,force,err)
                row=dict(force=force,group=mode,max_hdr=err)
                if mode:
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER,r.buffers[1])
                    count=int(np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,4),np.uint32)[0])
                    mask=r.surface((r.w,r.h)).read(mask=True)[::-1,:,0]>.5
                    expected=np.flatnonzero(mask.reshape(-1)).astype(np.uint32)
                    assert count==len(expected),('queue count',count,len(expected))
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER,r.buffers[0])
                    pixels=(np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,count*4),np.uint32)
                            if count else np.array([],np.uint32))
                    assert np.array_equal(np.sort(pixels),expected),'Queue lost or duplicated pixels'
                    row['queued_pixels']=count
                queue_checks.append(row)
        result['queue_checks']=queue_checks
        for kernel in ('opaque','lazy'):
            r.scene(**cases[0]); a,_=r.capture(**variant,kernel=kernel)
            r.scene(**cases[1]); r.capture(**variant,kernel=kernel)
            r.scene(**cases[0]); b,_=r.capture(**variant,kernel=kernel)
            assert np.array_equal(a,b),('history',kernel)
        result['A_B_A_bit_identical']=True
        assert glGetError()==GL_NO_ERROR
    finally:
        r.close()
    from single_frame_v2 import SingleFrameV2Renderer
    from benchmark import scenes, set_scene
    result['selected_checks']=[]
    for width,height in ((801,451),(160,90),(2560,1440)):
        r=SingleFrameV2Renderer(width,height)
        try:
            selected=scenes() if width==2560 else [dict(c,name='case%d'%i,sky_only=False) for i,c in enumerate(cases)]
            for scene in selected:
                set_scene(r,scene)
                # V2 first: catches cold allocation/binding problems too.
                v2,_=r.capture(v2=True)
                v1,_=r.capture(optimized=True)
                identical=bool(np.array_equal(v1,v2))
                maximum=float(np.abs(v1-v2).max())
                rmse=float(np.sqrt(np.mean((display_color(v1)-display_color(v2))**2)))
                assert rmse<1e-5,('selected V2 parity',scene['name'],width,rmse)
                result['selected_checks'].append(dict(scene=scene['name'],resolution=[width,height],
                    bit_identical=identical,max_hdr=maximum,display_rmse=rmse))
            assert glGetError()==GL_NO_ERROR
        finally:
            r.close()
    result['pass']=True
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
