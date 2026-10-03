"""Interleaved GPU timing, image-quality sweep, spatial and temporal diagnostics."""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from cascade import Renderer, display_color, HERE

ROOT = HERE.parent
# Only import the established camera definitions; never write into Paper 0.
sys.path.insert(0, str(ROOT.parent/'0_cloudworks'/'program'))
from psms_gate import configs


def scenes():
    result = []
    for name, label, camera, bias, solar in configs():
        if solar is None:
            sun = (math.cos(3.5)*math.cos(-3.5), math.sin(3.5)*math.cos(-3.5), math.sin(-3.5))
        else:
            e, a = map(math.radians, solar)
            sun = (math.cos(e)*math.sin(a), math.cos(e)*math.cos(a), math.sin(e))
        result.append(dict(name=name, label=label, camera=camera, bias=bias,
                           sun=sun, sky_only=False, itime=10.0))
    for name, elevation in [('sky-day', 20), ('sky-twilight', -1)]:
        e = math.radians(elevation)
        result.append(dict(name=name, label='Atmosphere only', camera=(57500,22500,0),
                           bias=0.4, sun=(0,math.cos(e),math.sin(e)),
                           sky_only=True, itime=10.0))
    return result


def set_scene(renderer, scene):
    renderer.scene(**{k:v for k,v in scene.items() if k not in ('name','label')})


def variants(preset, factor):
    out = OrderedDict([('full', dict(levels=1))])
    settings = [(3, 0.005), (3, 0.01), (3, 0.02), (3, 0.04),
                (4, 0.005), (4, 0.01), (4, 0.02), (2, 0.01), (2, 0.02)]
    if preset == 'smoke':
        settings = [(3, 0.01), (3, 0.04)]
    elif preset == 'confirm':
        settings = [(3, 0.02), (3, 0.04), (3, 0.08), (3, 0.12),
                    (2, 0.04), (2, 0.08), (4, 0.04)]
    for levels, threshold in settings:
        out['L%d-T%g' % (levels, threshold)] = dict(
            levels=levels, factor=factor, threshold=threshold, radius=1)
    if preset == 'confirm':
        out['L3-T0.04-R0'] = dict(levels=3,factor=factor,threshold=0.04,radius=0)
    out['uniform-half'] = dict(levels=2, factor=factor, force=-1)
    out['uniform-quarter'] = dict(levels=3, factor=factor, force=-1)
    return out


def image_metrics(hdr, reference):
    a, b = display_color(hdr).astype(np.float64), display_color(reference).astype(np.float64)
    diff = a-b
    rmse = float(np.sqrt(np.mean(diff*diff)))
    lum = np.array([0.2126, 0.7152, 0.0722])
    ya, yb = a@lum, b@lum
    dx = np.diff(ya, axis=1)-np.diff(yb, axis=1)
    dy = np.diff(ya, axis=0)-np.diff(yb, axis=0)
    edge = math.sqrt(float((np.mean(dx*dx)+np.mean(dy*dy))/2))
    gx, gy = np.gradient(yb)
    edge_mask = np.hypot(gx, gy) >= max(0.01, float(np.percentile(np.hypot(gx,gy), 90)))
    edge_region = float(np.sqrt(np.mean(diff[edge_mask]**2))) if edge_mask.any() else 0.0
    ma, mb = gaussian_filter(ya, 1.5), gaussian_filter(yb, 1.5)
    va = np.maximum(gaussian_filter(ya*ya, 1.5)-ma*ma, 0)
    vb = np.maximum(gaussian_filter(yb*yb, 1.5)-mb*mb, 0)
    cov = gaussian_filter(ya*yb, 1.5)-ma*mb
    ssim = ((2*ma*mb+0.01**2)*(2*cov+0.03**2))/((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2))
    worst_channel = np.max(np.abs(diff), axis=2)
    return dict(rmse=rmse, psnr_db=-20*math.log10(max(rmse,1e-12)),
                ssim_luma=float(ssim.mean()), edge_gradient_rmse=edge,
                edge_region_rmse=edge_region, p99_abs_rgb=float(np.percentile(worst_channel,99)),
                bad_pixel_fraction_003=float((worst_channel>0.03).mean()),
                max_abs_rgb=float(worst_channel.max()),
                hdr_relative_rmse=float(np.sqrt(np.mean((hdr[...,:3]-reference[...,:3])**2)) /
                    max(float(np.sqrt(np.mean(reference[...,:3]**2))), 1e-12)))


def save_png(path, display):
    Image.fromarray(np.uint8(np.clip(display,0,1)*255+0.5)).save(path)


def contact_sheet(path, rows):
    w, h = 400, 225
    sheet = Image.new('RGB', (4*w, len(rows)*(h+25)), (18,18,22))
    draw = ImageDraw.Draw(sheet)
    for i,(label,ref,out,error,mask) in enumerate(rows):
        for j,(title,array) in enumerate([('full',ref), ('cascade',out), ('error x8',error), ('fine mask',mask)]):
            img = Image.fromarray(np.uint8(np.clip(array,0,1)*255+0.5))
            img = img.resize((w,h), Image.Resampling.LANCZOS)
            sheet.paste(img, (j*w, i*(h+25)+25))
            draw.text((j*w+4,i*(h+25)+5), label+' | '+title, fill=(235,235,235))
    sheet.save(path)


def summary(records, variant_defs):
    result = {}
    base_total = sum(r['variants']['full']['median_ms'] for r in records)
    for key in variant_defs:
        vs = [r['variants'][key] for r in records]
        ms = sum(v['median_ms'] for v in vs)
        result[key] = dict(mean_ms=ms/len(vs), aggregate_speedup=base_total/ms,
            arithmetic_mean_scene_speedup=float(np.mean([v['speedup'] for v in vs])),
            min_scene_speedup=min(v['speedup'] for v in vs),
            mean_rmse=float(np.mean([v['quality']['rmse'] for v in vs])),
            worst_rmse=max(v['quality']['rmse'] for v in vs),
            min_ssim=min(v['quality']['ssim_luma'] for v in vs),
            worst_bad_pixel_fraction=max(v['quality']['bad_pixel_fraction_003'] for v in vs),
            mean_shading_fraction=float(np.mean([v['work']['full_resolution_equivalent_shading'] for v in vs])),
            quality_gate_pass=all(v['quality']['rmse']<=0.005 and
                v['quality']['ssim_luma']>=0.99 and
                v['quality']['bad_pixel_fraction_003']<=0.01 for v in vs))
        if all('quality_vs_anchor' in v for v in vs):
            result[key]['quality_vs_anchor_pass']=all(v['quality_vs_anchor']['pass'] for v in vs)
        result[key]['no_measured_slowdown']=all(v['speedup']>=1.0 for v in vs)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=450)
    parser.add_argument('--frames', type=int, default=11)
    parser.add_argument('--factor', type=int, default=2)
    parser.add_argument('--preset', choices=['smoke','sweep','confirm'], default='sweep')
    parser.add_argument('--scenes', help='comma-separated scene IDs')
    parser.add_argument('--output', default='sweep_800')
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Run GPU correctness checks and exit')
    parser.add_argument('--profile', action='store_true', help='Record draw-only pass times in addition to frame times')
    parser.add_argument('--visual-variant', default=None)
    parser.add_argument('--levels', type=int, help='Custom cascade level count, replacing the preset sweep')
    parser.add_argument('--threshold', type=float, default=0.04, help='Threshold for --levels custom run')
    parser.add_argument('--radius', type=int, choices=[0,1,2], default=1, help='Parent-pixel halo for --levels')
    parser.add_argument('--research', action='store_true', help='Shared-work research variants')
    parser.add_argument('--experiment', choices=['shared','curvature','hash','roof','candidates','chunks','bounded','lighting','separable','transport','split','selected'], default='shared')
    parser.add_argument('--variant-keys', help='Comma-separated research variant subset; always includes full')
    parser.add_argument('--heldout', action='store_true', help='Deterministic unseen camera/time/sun perturbations')
    args = parser.parse_args()
    if args.frames < 3:
        parser.error('At least 3 timed frames required')
    out = ROOT/'results'/args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        from verify import run
        run(out)
        return
    defs = variants(args.preset, args.factor)
    if args.factor < 2:
        parser.error('--factor must be >= 2')
    if args.levels is not None:
        if args.levels < 2 or args.threshold < 0:
            parser.error('Custom --levels >= 2 and --threshold >= 0 required')
        defs = OrderedDict([('full',dict(levels=1)),('custom',dict(
            levels=args.levels,factor=args.factor,threshold=args.threshold,radius=args.radius))])
    selected = scenes()
    if args.scenes:
        requested = args.scenes.split(',')
        selected = [s for s in selected if s['name'] in requested]
        if len(selected) != len(set(requested)):
            parser.error('Unknown scene name')
    elif args.preset == 'smoke':
        selected = [s for s in selected if s['name'] in ('A-g','B-t','postset-side','sky-day')]
    if args.heldout:
        rng=np.random.default_rng(90210)
        selected=[dict(s) for s in selected if not s['sky_only']]
        for s in selected:
            s['name']='heldout-'+s['name']
            s['camera']=tuple(np.asarray(s['camera'])+rng.uniform([-1200,-1200,-50],[1200,1200,50]))
            s['itime']=float(rng.uniform(25,85))
            az=math.atan2(s['sun'][0],s['sun'][1])+math.radians(float(rng.uniform(-12,12)))
            el=math.asin(s['sun'][2])+math.radians(float(rng.uniform(-2,2)))
            s['sun']=(math.cos(el)*math.sin(az),math.cos(el)*math.cos(az),math.sin(el))
            s['bias']+=float(rng.uniform(-.06,.06))
    if args.research:
        if args.experiment=='selected':
            from single_frame import SingleFrameRenderer
            renderer=SingleFrameRenderer(args.width,args.height)
        else:
            from fast import FastRenderer
            renderer = FastRenderer(args.width, args.height)
        defs = OrderedDict([('full',dict(levels=1)),
            ('shared-full',dict(shared=True,levels=1)),
            ('shared-cascade',dict(shared=True,levels=3,threshold=args.threshold)),
            ('lookup-full',dict(lookup=True,levels=1)),
            ('lookup-cascade',dict(lookup=True,levels=3,threshold=args.threshold))])
        if args.experiment == 'curvature':
            defs = OrderedDict([('full',dict(levels=1)),
                ('lookup-cascade',dict(lookup=True,levels=3,threshold=0.04))])
            for t in (.003,.006,.012):
                defs['curvature-%g'%t] = dict(lookup=True,curvature=True,levels=3,threshold=t)
        if args.experiment == 'hash':
            defs = OrderedDict([('full',dict(levels=1)),
                ('hash-full',dict(hashed=True,levels=1)),
                ('hash-cascade',dict(hashed=True,levels=3,threshold=.04)),
                ('hash-curvature',dict(hashed=True,curvature=True,levels=3,threshold=.006))])
        if args.experiment == 'roof':
            defs = OrderedDict([('full',dict(levels=1))])
            for size in (1024,2048):
                defs['roof%d-full'%size] = dict(roof=size,levels=1)
                defs['roof%d-cascade'%size] = dict(roof=size,levels=3,threshold=.04)
        if args.experiment == 'candidates':
            defs = OrderedDict([('full',dict(levels=1)),
                ('candidates-full',dict(candidates=True,levels=1)),
                ('candidates-L3',dict(candidates=True,levels=3,threshold=.04)),
                ('candidates-L4',dict(candidates=True,levels=4,threshold=.04)),
                ('candidates-N4',dict(candidates=True,levels=2,factor=4,threshold=.04))])
        if args.experiment == 'chunks':
            defs = OrderedDict([('full',dict(levels=1)),
                ('chunks-full',dict(chunks=True,levels=1)),
                ('chunks-N4',dict(chunks=True,levels=2,factor=4,threshold=.04))])
        if args.experiment == 'bounded':
            defs = OrderedDict([('full',dict(levels=1)),
                ('bounded-full',dict(bounded=True,levels=1)),
                ('bounded-N4',dict(bounded=True,scores=True,levels=2,factor=4,threshold=.04)),
                ('bounded-N4-T06',dict(bounded=True,scores=True,levels=2,factor=4,threshold=.06)),
                ('bounded-N8',dict(bounded=True,scores=True,levels=2,factor=8,threshold=.04))])
        if args.experiment == 'lighting':
            defs=OrderedDict([('full',dict(levels=1))])
            for mode in (3,4,10):
                defs['lighting%d-full'%mode]=dict(lighting=mode,levels=1)
                defs['lighting%d-N4'%mode]=dict(lighting=mode,scores=True,levels=2,factor=4,threshold=.03)
        if args.experiment == 'separable':
            defs=OrderedDict([('full',dict(levels=1))])
            for n,t in ((4,.025),(4,.04),(8,.025),(8,.04)):
                defs['separable%d-T%g'%(n,t)]=dict(separable=n,threshold=t)
        if args.experiment == 'transport':
            defs=OrderedDict([('full',dict(levels=1)),('bounded-N4',dict(bounded=True,scores=True,levels=2,factor=4,threshold=.04))])
            for t in (.02,.05,.1):
                defs['transport-%g'%t]=dict(reuse_light=t,scores=True,levels=2,factor=4,threshold=.035)
        if args.experiment == 'split':
            defs=OrderedDict([('full',dict(levels=1)),('bounded-N4',dict(bounded=True,scores=True,levels=2,factor=4,threshold=.04))])
            for t in (.05,.1,.2):
                defs['split-%g'%t]=dict(split_light=t,levels=2,factor=4,threshold=.035)
        if args.experiment == 'selected':
            defs=OrderedDict([('full',dict(levels=1)),
                ('anchor',dict(levels=3,factor=2,threshold=.04)),
                ('optimized',dict(optimized=True,levels=2,factor=4,threshold=.04)),
                ('aggressive',dict(optimized=True,levels=3,factor=2,threshold=.12))])
    else:
        renderer = Renderer(args.width, args.height)
    if args.variant_keys:
        requested = ['full'] + args.variant_keys.split(',')
        if any(k not in defs for k in requested):
            parser.error('Unknown variant key')
        defs = OrderedDict((k,v) for k,v in defs.items() if k in requested)
    print('GPU:', renderer.gpu, '|', args.width,args.height, flush=True)
    metadata = dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), gpu=renderer.gpu,
        opengl=renderer.version, resolution=[args.width,args.height], frames=args.frames,
        factor_semantics='N per axis, N^2 pixels', variant_definitions=defs,
        reference='same PSMS Next_2 shader shaded at every final-resolution pixel; NOT converged transport',
        timing='GL_TIME_ELAPSED entire cascade, rotated variant order each repetition; excludes readback',
        quality_gate=dict(max_rmse=0.005,min_ssim=0.99,max_bad_pixel_fraction_003=0.01),
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [HERE/'baseline_psms.glsl',HERE/'cascade.py',HERE/'reconstruct.frag',HERE/'stencil.frag',Path(__file__)]})
    records, sheet = [], []
    visual_key = args.visual_variant or ('optimized' if args.research and args.experiment=='selected' else 'shared-cascade' if args.research else 'custom' if args.levels else
                  ('L3-T0.04' if args.preset=='confirm' else 'L3-T0.01'))
    if visual_key not in defs:
        parser.error('Unknown visual variant')
    metadata['visual_variant'] = visual_key
    metadata['python'] = sys.version
    metadata['numpy'] = np.__version__
    if args.research:
        metadata['hashes']['fast.py'] = hashlib.sha256((HERE/'fast.py').read_bytes()).hexdigest()
        metadata['hashes']['curvature.frag'] = hashlib.sha256((HERE/'curvature.frag').read_bytes()).hexdigest()
        metadata['hashes']['chunk_cache.py'] = hashlib.sha256((HERE/'chunk_cache.py').read_bytes()).hexdigest()
        for filename in ('parent_score.frag','scored_reconstruct.frag','axis_reconstruct.frag','transport.py',
                         'transport_reconstruct.frag','transport_stencil.frag'):
            metadata['hashes'][filename]=hashlib.sha256((HERE/filename).read_bytes()).hexdigest()
        if args.experiment=='selected':
            for filename in ('single_frame.py','optimized_sources.py'):
                metadata['hashes'][filename]=hashlib.sha256((HERE/filename).read_bytes()).hexdigest()
            metadata['selected_policy']='N4 two levels, T=.04; original single pass below 800*450 pixels'
    metadata['heldout']=args.heldout
    # Keep the code actually measured even if the research implementation evolves.
    (out/'source').mkdir(exist_ok=True)
    for filename in metadata['hashes']:
        shutil.copyfile(HERE/filename,out/'source'/filename)
    try:
        for scene in selected:
            set_scene(renderer,scene)
            # Create storage and execute all programs BEFORE measured repetitions.
            for variant in defs.values():
                renderer.render(**variant)
            renderer.warm(1.0)
            timings = {k:[] for k in defs}
            keys = list(defs)
            for rep in range(args.frames):
                # Rotating forward/reverse ordering reduces sustained ordering bias.
                order = keys[rep%len(keys):]+keys[:rep%len(keys)]
                if rep%2:
                    order = order[::-1]
                for key in order:
                    timings[key].append(renderer.timed(**defs[key]))
            reference, base_work = renderer.capture()
            base_ms = float(np.median(timings['full']))
            ref_disp = display_color(reference)
            save_png(out/(scene['name']+'_full.png'),ref_disp)
            variants_result = {}
            for key,variant in defs.items():
                if key == 'full':
                    img, work = reference, base_work
                else:
                    img, work = renderer.capture(**variant)
                quality = image_metrics(img,reference)
                ms = float(np.median(timings[key]))
                variants_result[key] = dict(median_ms=ms,
                    p95_ms=float(np.percentile(timings[key],95)), samples_ms=timings[key],
                    speedup=base_ms/ms, quality=quality, work=work)
                if key in ('anchor','aggressive'):
                    save_png(out/(scene['name']+'_'+key+'.png'),display_color(img))
                if args.profile and key in ('full',visual_key):
                    profiles = [renderer.profile(**variant) for _ in range(args.frames)]
                    variants_result[key]['draw_only_profile_ms'] = {
                        stage:float(np.median([p[stage] for p in profiles])) for stage in profiles[0]}
                if key == visual_key:
                    d = display_color(img)
                    mask = renderer.surface((args.width,args.height)).read(mask=True)
                    mask = np.repeat(mask,3,axis=2)
                    error = np.abs(d-ref_disp)*8
                    save_png(out/(scene['name']+'_cascade.png'),d)
                    save_png(out/(scene['name']+'_error_x8.png'),error)
                    save_png(out/(scene['name']+'_mask.png'),mask)
                    # Keep the contact sheet small; native-size PNGs above retain details.
                    previews = []
                    for a in (ref_disp,d,error,mask):
                        im = Image.fromarray(np.uint8(np.clip(a,0,1)*255+0.5))
                        previews.append(np.asarray(im.resize((400,225),Image.Resampling.LANCZOS))/255.)
                    sheet.append((scene['name'],*previews))
            if 'anchor' in variants_result:
                anchor=variants_result['anchor']['quality']
                for value in variants_result.values():
                    q=value['quality']
                    checks=dict(rmse=q['rmse']<=anchor['rmse']+1e-6,
                        ssim=q['ssim_luma']>=anchor['ssim_luma']-1e-6,
                        bad_pixels=q['bad_pixel_fraction_003']<=anchor['bad_pixel_fraction_003']+1e-6)
                    value['quality_vs_anchor']=dict(checks,pass_=all(checks.values()))
                    value['quality_vs_anchor']['pass']=value['quality_vs_anchor'].pop('pass_')
            if args.temporal:
                # Compare the change of the approximation error, so genuine scene
                # motion does not get mislabeled as reconstruction flicker.
                temporal = {}
                for mode in ('time','camera'):
                    other = dict(scene)
                    if mode == 'time':
                        other['itime'] += 1/60
                    else:
                        other['camera'] = tuple(np.array(scene['camera'])+np.array([10,0,0]))
                    set_scene(renderer,other)
                    ref2,_ = renderer.capture()
                    set_scene(renderer,scene)
                    for key in (k for k in defs if k not in ('full','uniform-half','uniform-quarter')):
                        first,_ = renderer.capture(**defs[key])
                        set_scene(renderer,other)
                        second,_ = renderer.capture(**defs[key])
                        e = (display_color(second)-display_color(ref2))-(display_color(first)-ref_disp)
                        temporal.setdefault(key,{})[mode+'_error_delta_rmse'] = float(np.sqrt(np.mean(e*e)))
                        set_scene(renderer,scene)
                for key, v in temporal.items():
                    variants_result[key]['temporal'] = v
            record = dict(scene=scene, variants=variants_result)
            records.append(record)
            print(scene['name'], 'full %.3f ms' % base_ms, flush=True)
            for k,v in variants_result.items():
                if k != 'full':
                    print('  %-17s %7.3f ms %5.2fx rmse %.6f SSIM %.5f shade %.1f%%' %
                        (k,v['median_ms'],v['speedup'],v['quality']['rmse'],v['quality']['ssim_luma'],
                         100*v['work']['full_resolution_equivalent_shading']),flush=True)
            payload = dict(metadata=metadata,records=records,summary=summary(records,defs))
            cloud_records=[r for r in records if not r['scene']['sky_only']]
            if cloud_records:
                payload['cloud_summary']=summary(cloud_records,defs)
            (out/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n')
        contact_sheet(out/'comparison.png',sheet)
        print('Summary:',json.dumps(payload['summary'],indent=2),flush=True)
    finally:
        renderer.close()


if __name__ == '__main__':
    main()
