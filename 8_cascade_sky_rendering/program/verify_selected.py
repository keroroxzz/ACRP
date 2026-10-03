"""Independent field, interpolation, history, and fallback checks on real GPU."""
import json
import hashlib
import shutil
import numpy as np
from OpenGL.GL import *
from cascade import Renderer, Surface, compile_program, sky_source, display_color, HERE
from single_frame import SingleFrameRenderer
from verify import bilinear


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    r=SingleFrameRenderer(801,451)
    cases=[dict(camera=(25000,0,0),bias=.4,sun=(0,.9998477,.0174524),itime=10),
           dict(camera=(-43000,17500,1200),bias=-.4,sun=(.6,.7998,.01745),itime=37.9),
           dict(camera=(250000,-330000,2700),bias=-.5,sun=(0,.9998477,-.0174524),itime=83.3)]
    result={'gpu':r.gpu,'resolution':[r.w,r.h],'cases':[]}
    (out/'source').mkdir(exist_ok=True)
    result['hashes']={}
    for filename in ('verify_selected.py','verify.py','single_frame.py','optimized_sources.py',
                     'cascade.py','baseline_psms.glsl','parent_score.frag','scored_reconstruct.frag',
                     'reconstruct.frag','stencil.frag'):
        result['hashes'][filename]=hashlib.sha256((HERE/filename).read_bytes()).hexdigest()
        shutil.copyfile(HERE/filename,out/'source'/filename)
    try:
        for scene in cases:
            r.scene(**scene)
            ref,_=r.capture()
            full,_=r.capture(optimized=True,levels=1)
            error=float(np.sqrt(np.mean((display_color(full)-display_color(ref))**2)))
            # The gather LUT is an approximation: record its total image error,
            # rather than claiming bit parity with the analytic reference.
            assert np.isfinite(error)
            forced,work=r.capture(optimized=True,levels=2,factor=4,force=1)
            maximum=float(np.abs(forced-full).max())
            assert maximum<1e-6,('forced full parity',maximum)
            interpolated,work=r.capture(optimized=True,levels=2,factor=4,force=-1)
            base=r.surface((201,113)).read().astype(np.float64)
            expected=bilinear(base,(r.w,r.h))
            maximum_interp=float(np.abs(interpolated-expected).max())
            display_interp=float(np.abs(display_color(interpolated)-display_color(expected)).max())
            # HDR sun radiance magnifies float32 coordinate rounding; compare
            # against the float64 CPU oracle in the actual display domain too.
            assert display_interp<1e-4,('CPU bilinear display parity',display_interp)
            result['cases'].append(dict(scene=scene,full_display_rmse=error,
                force_full_max_hdr=maximum,force_interpolate_max_hdr=maximum_interp,
                force_interpolate_max_display=display_interp))

        field=Surface(501,503)
        ending='''
void main() {
    vec2 p=uCamera.xy+(gl_FragCoord.xy/vec2(501.0,503.0)*2.0-1.0)*45000.0;
    float h=HSELowCloudTop(p,iTime*100.0+5000.0,0.0);
    outColor=vec4(h,0.0,0.0,1.0);
}
'''
        programs=[compile_program(s[:s.index('void mainImage')]+ending)
                  for s in (sky_source(),r.optimized_source)]
        field_errors=[]
        for scene in cases:
            r.scene(**scene)
            r.render(optimized=True,levels=1)  # rebuild this image's tables
            images=[]
            for p in programs:
                r.sky=p
                Renderer.scene(r,**scene)
                glUniform1i(r.loc(p,'uEvents'),5)
                glUniform1i(r.loc(p,'uCandidates'),7)
                field.bind()
                glDrawBuffers(1,[GL_COLOR_ATTACHMENT0])
                glDisable(GL_STENCIL_TEST)
                r.triangle()
                images.append(field.read())
            error=float(np.abs(images[0]-images[1]).max())
            assert error<.02,('conservative event field parity, metres',error)
            field_errors.append(error)
        for p in programs:
            glDeleteProgram(p)
        field.close()
        r.sky=r.original_sky
        result['roof_field_max_error_metres']=field_errors
        result['roof_field_samples_per_case']=501*503

        variant=dict(optimized=True,levels=2,factor=4,threshold=.04)
        r.scene(**cases[0]); first,_=r.capture(**variant)
        r.scene(**cases[1]); r.capture(**variant)
        r.scene(**cases[0]); repeated,_=r.capture(**variant)
        assert np.array_equal(first,repeated),'History dependence'
        result['A_B_A_bit_identical']=True
        sky=dict(cases[0],sky_only=True,sun=(0,.9998477,-.0174524))
        r.scene(**sky)
        sky_ref,_=r.capture()
        sky_opt,work=r.capture(optimized=True)
        sky_anchor,_=r.capture(levels=3,factor=2,threshold=.04)
        sky_error=float(np.sqrt(np.mean((display_color(sky_opt)-display_color(sky_ref))**2)))
        anchor_error=float(np.sqrt(np.mean((display_color(sky_anchor)-display_color(sky_ref))**2)))
        assert sky_error<=anchor_error,('full resolution twilight sky vs anchor',sky_error,anchor_error)
        assert work['full_resolution_sky']
        result['twilight_sky_display_rmse']=sky_error
        result['twilight_anchor_display_rmse']=anchor_error
        assert glGetError()==GL_NO_ERROR
    finally:
        r.close()
    r=SingleFrameRenderer(160,90)
    try:
        r.scene(**cases[0])
        ref,_=r.capture()
        fallback,work=r.capture(optimized=True,levels=2,factor=4)
        assert np.array_equal(ref,fallback)
        assert work['small_resolution_fallback']
        result['small_resolution_fallback_bit_identical']=True
        assert glGetError()==GL_NO_ERROR
    finally:
        r.close()
    result['pass']=True
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
