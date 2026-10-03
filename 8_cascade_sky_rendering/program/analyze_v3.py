"""Offline V3 RMSE budgets, frozen selection, validation and native image viewer."""
import argparse
import csv
import json
import math
import hashlib
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v3 import RMSE_BUDGETS

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'results'
SWEEP=RESULTS/'v3_sweep_1440'
HOLDOUT=RESULTS/'v3_heldout_1440'
SELECTION=SWEEP/'selection.json'


def read(folder):
    return json.loads((folder/'metrics.json').read_text())


def choose(data):
    assert len(data['records'])==14,'Incomplete calibration sweep'
    candidates={k:v for k,v in data['summary'].items() if k not in ('full','anchor')}
    tiers=[]
    for cap in RMSE_BUDGETS:
        allowed=[k for k,v in candidates.items() if v['worst_rmse']<=cap]
        key=min(allowed,key=lambda k:candidates[k]['mean_ms']) if allowed else None
        tiers.append(dict(rmse_cap=cap,variant=key))
    ten=[k for k,v in candidates.items() if v['aggregate_speedup']>=10]
    first_ten=min(ten,key=lambda k:candidates[k]['worst_rmse']) if ten else None
    keys=list(dict.fromkeys(['anchor','v2']+[t['variant'] for t in tiers if t['variant']]+([first_ten] if first_ten else [])))
    return dict(policy='Fastest FIXED candidate with every calibration scene RMSE <= cap; no runtime reference access',
                calibration_scenes=14,rmse_domain='float display RGB in [0,1], before PNG quantization',
                tiers=tiers,lowest_error_10x_candidate=first_ten,validation_keys=keys,
                frozen_definitions={k:data['metadata']['variant_definitions'][k] for k in keys},
                validation_seed=20261004)


def pareto(data):
    vals={k:v for k,v in data['summary'].items() if k not in ('full','anchor')}
    return [k for k,a in vals.items() if not any(b['mean_ms']<=a['mean_ms'] and b['worst_rmse']<=a['worst_rmse'] and
               (b['mean_ms']<a['mean_ms'] or b['worst_rmse']<a['worst_rmse']) for j,b in vals.items() if j!=k)]


def plots(data,selection):
    fig,ax=plt.subplots(figsize=(9,5))
    for prefix,label,color in [('L2-','2 levels / N4','#267dae'),('L3-','3 levels / N2','#d67b24')]:
        keys=[k for k in data['summary'] if k.startswith(prefix)]
        xs=[data['summary'][k]['worst_rmse'] for k in keys]
        ys=[data['summary'][k]['aggregate_speedup'] for k in keys]
        ax.plot(xs,ys,'o-',label=label,color=color)
        for k,x,y in zip(keys,xs,ys):
            ax.annotate(k.split('-T')[1],(x,y),xytext=(4,5),textcoords='offset points',fontsize=8)
    a=data['summary']['v2'];ax.scatter([a['worst_rmse']],[a['aggregate_speedup']],s=100,marker='*',color='black',label='V2 T=.035')
    ax.axhline(10,ls='--',color='#777',label='10x target')
    ax.set_title('RTX 4070 | 2560x1440 | 14 scenes | labels = gradient threshold T',fontsize=10)
    ax.set_xlabel('Worst scene display RGB RMSE (lower is better)')
    ax.set_ylabel('Sum of full-reference times / sum of candidate times')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout()
    fig.savefig(SWEEP/'quality_speed.png',dpi=180);fig.savefig(SWEEP/'quality_speed.svg');plt.close(fig)


def crop_sheet(data,selection):
    keys=['full','v2']
    for cap in (.003,.005,.010,.015):
        k=next(t['variant'] for t in selection['tiers'] if t['rmse_cap']==cap)
        if k and k not in keys:keys.append(k)
    probe=selection['lowest_error_10x_candidate'] or keys[-1]
    names=['A-g','A-t','B-t','B-h','preset-front','postset-side']
    side=320;row_h=side+36
    sheet=Image.new('RGB',(side*len(keys),row_h*len(names)),(20,21,25))
    draw=ImageDraw.Draw(sheet);coords={}
    for row,name in enumerate(names):
        a=np.asarray(Image.open(SWEEP/(name+'_full.png')),dtype=np.float32)/255
        b=np.asarray(Image.open(SWEEP/(name+'_'+probe+'.png')),dtype=np.float32)/255
        scores=uniform_filter(np.mean((a-b)**2,axis=2),size=side,mode='constant')
        half=side//2;inner=scores[half:-half,half:-half]
        y,x=map(int,np.unravel_index(np.argmax(inner),inner.shape))
        coords[name]=dict(x=x,y=y,width=side,height=side,selection_image=probe)
        for col,k in enumerate(keys):
            with Image.open(SWEEP/(name+'_'+k+'.png')) as im:
                sheet.paste(im.crop((x,y,x+side,y+side)),(col*side,row*row_h+36))
            draw.text((col*side+4,row*row_h+3),name+' | '+k,fill='white')
            q=next(r for r in data['records'] if r['scene']['name']==name)['variants'][k]
            draw.text((col*side+4,row*row_h+18),'scene RMSE %.5f | %.2fx'%(q['quality']['rmse'],q['speedup']),fill='white')
    sheet.save(SWEEP/'native_crops.png')
    (SWEEP/'crop_coordinates.json').write_text(json.dumps(dict(method='Maximum 320x320 local MSE of the lowest-error >=10x candidate, from quantized PNGs; same box for all variants',columns=keys,scenes=coords),indent=2)+'\n')


def bootstrap(data,key):
    rng=np.random.default_rng(309);n=3000
    totals={k:np.zeros(n) for k in ('full','v2',key)}
    for r in data['records']:
        for k in totals:
            a=np.asarray(r['variants'][k]['samples_ms'])
            totals[k]+=np.median(rng.choice(a,(n,len(a)),replace=True),axis=1)
    return {k:np.percentile(totals[k]/totals[key],[2.5,50,97.5]).tolist() for k in ('full','v2')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--select',action='store_true');args=parser.parse_args()
    data=read(SWEEP)
    if args.select:
        if SELECTION.exists():
            raise RuntimeError('Selection already frozen; use a new results directory to retune')
        selection=choose(data)
        SELECTION.write_text(json.dumps(selection,indent=2)+'\n')
        print('Frozen validation keys:',','.join(selection['validation_keys']))
        print(json.dumps(selection['tiers'],indent=2))
        return
    selection=json.loads(SELECTION.read_text());held=read(HOLDOUT)
    assert len(data['records'])==14 and len(held['records'])==12
    assert held['metadata']['heldout_seed']==selection['validation_seed']
    for k,v in selection['frozen_definitions'].items():
        assert held['metadata']['variant_definitions'][k]==v
    rows=[];tier_rows=[]
    for dataset,d in [('calibration',data),('validation',held)]:
        for r in d['records']:
            for k,v in r['variants'].items():
                rows.append([dataset,r['scene']['name'],k,v['median_ms'],v['speedup'],r['variants']['v2']['median_ms']/v['median_ms'],
                             v['quality']['rmse'],v['quality']['ssim_luma'],v['quality']['bad_pixel_fraction_003']])
    for tier in selection['tiers']:
        cap,k=tier['rmse_cap'],tier['variant']
        if k is None:continue
        c=data['summary'][k];h=held['summary'][k]
        fails=[r['scene']['name'] for r in held['records'] if r['variants'][k]['quality']['rmse']>cap]
        tier_rows.append(dict(rmse_cap=cap,variant=k,calibration=c,validation=h,validation_failed_scenes=fails,
                              validation_pass=not fails,calibration_vs_v2=data['summary']['v2']['mean_ms']/c['mean_ms'],
                              validation_vs_v2=held['summary']['v2']['mean_ms']/h['mean_ms'],
                              calibration_uncertainty=bootstrap(data,k),validation_uncertainty=bootstrap(held,k)))
    result=dict(selection=selection,tiers=tier_rows,calibration_pareto=pareto(data))
    if selection['lowest_error_10x_candidate']:
        k=selection['lowest_error_10x_candidate']
        result['temporal_diagnostics']={key:{metric:max(r['variants'][key]['temporal'][metric] for r in held['records'])
            for metric in ('time_error_delta_rmse','camera_error_delta_rmse')} for key in ('v2',k)}
    (SWEEP/'rmse_tiers.json').write_text(json.dumps(result,indent=2)+'\n')
    with (SWEEP/'per_scene.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['dataset','scene','variant','median_ms','speedup_full','speedup_v2','rmse','ssim','bad_fraction_003']);writer.writerows(rows)
    with (SWEEP/'rmse_tiers.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['rmse_cap','variant','calibration_worst_rmse','calibration_ms','calibration_speedup_full','calibration_speedup_v2','validation_worst_rmse','validation_ms','validation_speedup_full','validation_pass'])
        for t in tier_rows:
            c,h=t['calibration'],t['validation'];writer.writerow([t['rmse_cap'],t['variant'],c['worst_rmse'],c['mean_ms'],c['aggregate_speedup'],t['calibration_vs_v2'],h['worst_rmse'],h['mean_ms'],h['aggregate_speedup'],t['validation_pass']])
    plots(data,selection);crop_sheet(data,selection)
    html=(ROOT/'program/v3_viewer.html').read_text()
    package=dict(calibration=data,validation=held,tiers=tier_rows,selection=selection)
    (SWEEP/'compare.html').write_text(html.replace('/*DATA*/null',json.dumps(package)),encoding='utf-8')
    report(data,held,result)
    archive=SWEEP/'analysis_source';archive.mkdir(exist_ok=True)
    provenance={}
    for filename in ('analyze_v3.py','v3.py','v3_viewer.html'):
        path=ROOT/'program'/filename
        shutil.copyfile(path,archive/filename)
        provenance[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
    for label,path in [('calibration_metrics',SWEEP/'metrics.json'),('validation_metrics',HOLDOUT/'metrics.json'),('frozen_selection',SELECTION)]:
        provenance[label]=hashlib.sha256(path.read_bytes()).hexdigest()
    (archive/'sha256.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(tiers=[dict(cap=t['rmse_cap'],key=t['variant'],speedup=t['calibration']['aggregate_speedup'],heldout_pass=t['validation_pass']) for t in tier_rows],first_10x=selection['lowest_error_10x_candidate']),indent=2))


def report(data,held,result):
    s=data['summary'];hs=held['summary'];sel=result['selection'];ten=sel['lowest_error_10x_candidate']
    lines=['# V3：放寬 RMSE 的品質與速度評估','',
      '本輪保留 V2 的 shader、密度與陰影積分，評估減少螢幕空間細化工作能換得多少速度。RMSE 是事後量測的品質門檻，**不是渲染器可直接保證的誤差上界**。所有候選均使用單張影像，不借用歷史結果。','',
      '[互動畫質／效能圖與原尺寸切換](results/v3_sweep_1440/compare.html) · [原尺寸局部比較](results/v3_sweep_1440/native_crops.png) · [RMSE 分級 CSV](results/v3_sweep_1440/rmse_tiers.csv) · [逐場景 CSV](results/v3_sweep_1440/per_scene.csv)','',
      '## 評估方式','',
      'RTX 4070，2560×1440，每設定七次交錯計時。原十四場景作設定篩選，其中十二張體積雲、兩張純天空；主要速度為基準總中位數耗時除以候選總中位數耗時。純天空保持 V2 完整解析度路徑。',
      '掃描兩層 N=4 與三層 N=2，梯度門檻 T=.05/.07/.09/.12/.16/.22/.32，再加 V2（兩層 N=4、T=.035）、完整逐像素基準與原品質錨點，共 17 設定。N 為每軸解析度倍率。',
      'RMSE 在 tone map 與 gamma 後的浮點 RGB [0,1] 計算，參考為完整逐像素 PSMS shader，並非 V2 圖像或 PNG。每級要求**十四個場景各自 RMSE 都不超標**，再選整體最快的固定設定；不為個別場景挑最有利門檻。',
      '選定後凍結設定，使用 seed=20261004 的十二組新相機／時間／太陽方向驗證。驗證資料不參與選擇；超標會列出，不回頭調參或換候選。','',
      '## 每種 RMSE 標準換得的速度','',
      f'同輪 V2 平均 **{s["v2"]["mean_ms"]:.2f} ms**，對逐像素基準 **{s["v2"]["aggregate_speedup"]:.2f}×**；最差場景 RMSE **{s["v2"]["worst_rmse"]:.6f}**。','',
      '| 每場景 RMSE 上限 | 最快固定設定 | 實際最差 RMSE | 平均 ms | 對完整基準 | 對 V2 | 最低 SSIM | 最差 >.03 像素比例 |',
      '|---:|---|---:|---:|---:|---:|---:|---:|']
    for t in result['tiers']:
        c=t['calibration'];lines.append(f'| {t["rmse_cap"]:.4f} | {t["variant"]} | {c["worst_rmse"]:.6f} | {c["mean_ms"]:.2f} | {c["aggregate_speedup"]:.2f}× | {t["calibration_vs_v2"]:.2f}× | {c["min_ssim"]:.5f} | {100*c["worst_bad_pixel_fraction"]:.2f}% |')
    lines+=['','同一設定可能符合數個 RMSE 級別；表格是離散掃描結果，未宣稱連續參數的全域最佳解。SSIM 和局部大誤差是輔助觀察值，此輪未把 V2 的原品質門檻暗中當成新的通過條件。原始 metrics.json 的 quality_gate_pass／quality_vs_anchor_pass 仍保留舊規則；本輪各 RMSE 級別的判定請看 rmse_tiers.json。','',
            '![品質與速度](results/v3_sweep_1440/quality_speed.png)','', '## 平均 10× 的代價','']
    if ten:
        v=s[ten];h=hs[ten];cloud=data['cloud_summary'][ten]
        worst=max(data['records'],key=lambda r:r['variants'][ten]['quality']['rmse'])['scene']['name']
        lines+=[f'達到 10× 的候選中，最差場景 RMSE 最低的是 **{ten}**：原十四場景 **{v["aggregate_speedup"]:.2f}×**、**{v["mean_ms"]:.2f} ms**，比 V2 快 **{s["v2"]["mean_ms"]/v["mean_ms"]:.2f}×**。只看十二張體積雲為 **{cloud["aggregate_speedup"]:.2f}×**；最慢的雲場景為 **{cloud["min_scene_speedup"]:.2f}×**，不是每個場景都達 10×。',
                f'其場景平均 RMSE={v["mean_rmse"]:.6f}；最差 RMSE={v["worst_rmse"]:.6f}（{worst}）、最低 SSIM={v["min_ssim"]:.5f}、最差大誤差像素比例={100*v["worst_bad_pixel_fraction"]:.2f}%。新場景實測 **{h["aggregate_speedup"]:.2f}×**，最差 RMSE={h["worst_rmse"]:.6f}。',
                '這是放寬品質後的 10× 候選，不能等同於原先「維持 V2 畫質的 10×」。你先前對 V2 的肉眼判斷也不能直接套用到更激進設定，請以原尺寸切換與局部圖確認。']
    else:lines+=['本輪掃描沒有候選達到平均 10×。']
    lines+=['','## 凍結設定的新場景驗證','','| RMSE 上限 | 固定設定 | 新場景最差 RMSE | 加速 | 最低場景加速 | RMSE 通過 |','|---:|---|---:|---:|---:|---|']
    for t in result['tiers']:
        h=t['validation'];status='12/12' if t['validation_pass'] else f'{12-len(t["validation_failed_scenes"])}/12；超標：'+', '.join(t['validation_failed_scenes'])
        lines.append(f'| {t["rmse_cap"]:.4f} | {t["variant"]} | {h["worst_rmse"]:.6f} | {h["aggregate_speedup"]:.2f}× | {h["min_scene_speedup"]:.2f}× | {status} |')
    lines+=['','[原十四場景量測](results/v3_sweep_1440/metrics.json) · [凍結設定](results/v3_sweep_1440/selection.json) · [獨立場景量測](results/v3_heldout_1440/metrics.json) · [完整分級與 bootstrap 區間](results/v3_sweep_1440/rse_PLACEHOLDER.json)','',
       '## 視覺判讀與限制','',
       '互動頁可以選場景、選設定，以完整基準或 V2 為對照，原尺寸切換／交替顯示。品質曲線的每個點可點擊，並同步顯示该場景的 RMSE、SSIM 和速度。比較圖與 PNG 只供視覺判讀，指標由未量化浮點結果計算。',
       'RMSE 是全部顯示像素的平均平方誤差再開根號；例如 0.005 約相當於每通道 1.28/255 的 RMS 差異，但局部雲邊的差異可能大得多，不能解讀為每像素最多差 1.28 階。',
       '局部圖以最低 RMSE 的 10× 候選對完整基準之局部均方誤差選取 320×320 區域，各設定使用相同座標且不縮放。它刻意顯示較容易出現差異的位置；座標及選取依據保存於 crop_coordinates.json。',
       '本輪局部圖的 A-t、B-t 中，RMSE 0.010 級已可看到細碎雲紋與邊緣變軟，0.015 級更明顯；平滑天空與部分地面視角差異小得多。這是對保存影像的觀察，並非使用者可接受程度的結論。',
       '提高 T 讓更多區域由粗層內插，通常犧牲細雲、雲邊與局部紋理。RMSE 平均所有像素與色彩通道，無法單獨保證肉眼不可分辨；相同 RMSE 也可能有不同空間誤差分布。原尺寸比較尚不是人眼盲測。',
       '全部前置計算、分類、內插、stencil 與 shading 納入 GPU 計時；首次編譯、配置、讀回和 CPU 指標除外。每個資料夾保存每次時間樣本、場景輸入、source snapshots 與 SHA-256。bootstrap 只描述本次計時樣本的不確定性，不代表跨硬體／驅動保證。',
       '程式：[設定掃描](program/v3.py)、[評估與報告](program/analyze_v3.py)、[互動頁模板](program/v3_viewer.html)。沒有改動 V2 的渲染實作或覆蓋 V2 結果。','',
       '## 重現','','```bash',
       'python 9_cascade_sky_rendering/program/benchmark.py --research --experiment v3 --width 2560 --height 1440 --frames 7 --save-variants --output v3_sweep_1440',
       'MPLCONFIGDIR=/tmp/cascade_sky_mpl python 9_cascade_sky_rendering/program/analyze_v3.py --select',
       '# 將 --select 輸出的 frozen validation keys 原樣放入 --variant-keys：',
       'python 9_cascade_sky_rendering/program/benchmark.py --research --experiment v3 --heldout --heldout-seed 20261004 --variant-keys '+','.join(sel['validation_keys'])+' --width 2560 --height 1440 --frames 7 --save-variants --temporal --output v3_heldout_1440',
       'MPLCONFIGDIR=/tmp/cascade_sky_mpl python 9_cascade_sky_rendering/program/analyze_v3.py','```','',
       '重新選擇設定時應使用新結果目录；分析程式拒絕覆寫已凍結的 selection.json。','']
    (ROOT/'V3_REPORT.md').write_text('\n'.join(lines).replace('rse_PLACEHOLDER','rmse_tiers').replace('该','該').replace('目录','目錄'),encoding='utf-8')


if __name__=='__main__':main()
