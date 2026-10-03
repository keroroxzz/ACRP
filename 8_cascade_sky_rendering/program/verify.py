"""GPU invariants and an explicit counterexample to gradient-only refinement."""
import json
from pathlib import Path
import sys

import numpy as np
from OpenGL.GL import *
from PIL import Image

from cascade import Renderer, compile_program, display_color, level_sizes, HERE


def bilinear(a, size):
    w,h = size
    sh,sw = a.shape[:2]
    x = (np.arange(w)+0.5)*sw/w-0.5
    y = (np.arange(h)+0.5)*sh/h-0.5
    ix,iy = np.floor(x).astype(int),np.floor(y).astype(int)
    fx,fy = x-ix,y-iy
    x0,x1 = np.clip(ix,0,sw-1),np.clip(ix+1,0,sw-1)
    y0,y1 = np.clip(iy,0,sh-1),np.clip(iy+1,0,sh-1)
    top = a[y0[:,None],x0]*(1-fx)[None,:,None]+a[y0[:,None],x1]*fx[None,:,None]
    bot = a[y1[:,None],x0]*(1-fx)[None,:,None]+a[y1[:,None],x1]*fx[None,:,None]
    return top*(1-fy)[:,None,None]+bot*fy[:,None,None]


def run(out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    r = Renderer(161,91)
    results = {'gpu':r.gpu, 'resolution':[161,91]}
    try:
        r.scene((57500,22500,0),0.4,(0,1,0))
        full,_ = r.capture()
        for factor in (2,3):
            refined,stats = r.capture(levels=3,factor=factor,force=1)
            error = float(np.abs(refined-full).max())
            assert error < 1e-6, ('All-refine differs',factor,error)
            assert all(s['shaded_fraction']==1 for s in stats['levels'])
            results['force_all_N%d_max_abs_hdr'%factor] = error
            interpolated,stats = r.capture(levels=3,factor=factor,force=-1)
            sizes = level_sizes(r.w,r.h,3,factor)
            expected = r.surface(sizes[0]).read().astype(np.float64)
            for size in sizes[1:]:
                expected = bilinear(expected,size)
            error = float(np.abs(interpolated-expected).max())
            assert error < 1e-4, ('Bilinear differs',factor,error)
            assert all(s['shaded_fraction']==0 for s in stats['levels'][1:])
            results['force_none_N%d_max_abs_hdr'%factor] = error

        # Independent original GLSL 330 wrapper vs our GLSL 430 / uniform wrapper.
        sys.path.insert(0,str(HERE.parent.parent/'0_cloudworks'/'program'))
        import cw_gl as G
        src = G.place((HERE/'baseline_psms.glsl').read_text(),(57500,22500,0),0.4)
        src = G.sun(src,0,0)
        original = G.program(src)
        target = r.surface((r.w,r.h))
        target.bind()
        glDisable(GL_STENCIL_TEST)
        glDrawBuffers(1,[GL_COLOR_ATTACHMENT0])
        glUseProgram(original)
        glUniform3f(glGetUniformLocation(original,'iResolution'),r.w,r.h,1)
        glUniform1f(glGetUniformLocation(original,'iTime'),10)
        glUniform4f(glGetUniformLocation(original,'iMouse'),0,0,0,0)
        r.triangle()
        original_image = target.read()
        e = display_color(original_image)-display_color(full)
        error = float(np.sqrt(np.mean(e*e)))
        assert error < 1e-4, ('Original-wrapper parity',error)
        results['original_glsl330_vs_wrapper_display_rmse'] = error
        glDeleteProgram(original)

        # A feature smaller than a coarse pixel falls between all base samples.
        # A zero-gradient ancestor cannot reveal it; this is an expected limitation.
        toy = compile_program('''#version 430 core
layout(early_fragment_tests) in;
uniform vec2 uTargetSize;
layout(location=0) out vec4 outColor;
void main() {
    vec2 finalPixel=gl_FragCoord.xy/uTargetSize*vec2(161,91);
    float spot = float(all(lessThan(abs(finalPixel-vec2(11.5,13.5)),vec2(0.3))));
    outColor=vec4(vec3(0.1+spot),1.0);
}
''')
        real = r.sky
        r.sky = toy
        a,_ = r.capture()
        b,stats = r.capture(levels=3,threshold=0.005)
        r.sky = real
        glDeleteProgram(toy)
        assert np.max(np.abs(a-b)) > 0.5
        assert all(s['shaded_fraction']==0 for s in stats['levels'][1:])
        results['unseen_feature_counterexample'] = dict(
            expected_failure=True,max_abs_hdr=float(np.abs(a-b).max()),work=stats,
            explanation='The base image misses a one-pixel feature; its zero gradient never requests refinement.')
        for name,image in [('unseen_full',a),('unseen_cascade',b)]:
            Image.fromarray(np.uint8(display_color(image)*255+0.5)).save(out/(name+'.png'))
        assert glGetError()==GL_NO_ERROR
        results['invariants_pass'] = True
        (out/'verification.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results,indent=2),flush=True)
    finally:
        r.close()

