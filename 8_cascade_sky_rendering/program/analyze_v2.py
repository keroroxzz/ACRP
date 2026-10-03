"""Reproduce the V2 report from saved whole-frame GPU measurements."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from analyze_single_frame import crops, viewer, uncertainty, plt

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'results'


def read(name):
    return json.loads((RESULTS/name/'metrics.json').read_text())


def main():
    folder=RESULTS/'v2_final_1440'
    data=read(folder.name)
    held=read('v2_heldout_1440')
    small=read('v2_confirm_800')
    verification=json.loads((RESULTS/'verify_v2'/'verification.json').read_text())
    assert len(data['records'])==14 and len(held['records'])==12 and len(small['records'])==14
    for d in (data,held,small):
        assert d['summary']['v2']['quality_vs_anchor_pass']
    assert verification['pass']
    ci=uncertainty(data)
    (folder/'bootstrap_summary.json').write_text(json.dumps(ci,indent=2)+'\n')
    # Separate paired V1/V2 speed estimate; never compare unrelated run means.
    rng=np.random.default_rng(731)
    totals={k:np.zeros(4000) for k in ('v1','v2')}
    for r in data['records']:
        for k in totals:
            a=np.asarray(r['variants'][k]['samples_ms'])
            totals[k]+=np.median(rng.choice(a,(4000,len(a)),replace=True),axis=1)
    paired_ci=np.percentile(totals['v1']/totals['v2'],[2.5,50,97.5]).tolist()
    (folder/'paired_v1_v2_bootstrap.json').write_text(json.dumps(dict(seed=731,samples=4000,
        ratio_percentiles_025_50_975=paired_ci),indent=2)+'\n')
    for name,d in ((folder.name,data),('v2_heldout_1440',held),('v2_confirm_800',small)):
        f=RESULTS/name
        viewer(f,d)
        html=(f/'compare.html').read_text()
        html=html.replace('按 1、2、3 切換完整基準、原階層與本次版本。','按 1、2、3、4 切換完整基準、原階層、V2、V1。')
        html=html.replace('3 本次版本</button>','3 V2</button><button data-mode="v1">4 V1</button>')
        html=html.replace("['1','2','3'].includes(e.key)","['1','2','3','4'].includes(e.key)")
        html=html.replace("['full','anchor','cascade'][Number(e.key)-1]","['full','anchor','cascade','v1'][Number(e.key)-1]")
        (f/'compare.html').write_text(html)
        png_equal={r['scene']['name']:bool(np.array_equal(np.asarray(Image.open(f/(r['scene']['name']+'_v1.png'))),
                                 np.asarray(Image.open(f/(r['scene']['name']+'_cascade.png'))))) for r in d['records']}
        (f/'v1_v2_png_parity.json').write_text(json.dumps(png_equal,indent=2)+'\n')
    crops(folder,['A-t','B-t','B-h','preset-front','postset-side'])
    fig,ax=plt.subplots(figsize=(12,5))
    names=[r['scene']['name'] for r in data['records']]
    x=np.arange(len(names))
    for i,(key,label,color) in enumerate((('full','Full reference','#777777'),('anchor','Original cascade','#cc9830'),
                                         ('v1','V1','#579ac6'),('v2','V2','#218968'))):
        ax.bar(x+(i-1.5)*.2,[r['variants'][key]['median_ms'] for r in data['records']],.2,label=label,color=color)
    ax.set_xticks(x,names,rotation=40,ha='right'); ax.set_ylabel('Whole-frame GPU median (ms)')
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); ax.legend(); fig.tight_layout()
    fig.savefig(folder/'per_scene.png',dpi=160); plt.close(fig)
    rows=[]
    for p in sorted(RESULTS.glob('v2_*/metrics.json')):
        d=json.loads(p.read_text())
        for key,value in d['summary'].items():
            if key=='full': continue
            rows.append([p.parent.name,len(d['records']),key,value['mean_ms'],value['aggregate_speedup'],
                         d['summary'].get('v1',{'mean_ms':0})['mean_ms']/value['mean_ms'],
                         value['quality_vs_anchor_pass'],value['min_scene_speedup']])
    with (RESULTS/'v2_experiments.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['dataset','scenes','variant','mean_ms','speedup_vs_full',
                                              'speedup_vs_v1_same_run','quality_vs_anchor','min_speedup_vs_full']); writer.writerows(rows)
    s=data['summary']; h=held['summary']; m=small['summary']
    v1=s['v1']; v2=s['v2']; improvement=v1['mean_ms']/v2['mean_ms']
    reduction=1-v2['mean_ms']/v1['mean_ms']
    failed_v1=[r['scene']['name'] for r in data['records'] if r['variants']['v2']['median_ms']>r['variants']['v1']['median_ms']]
    identical=sum(v['bit_identical'] for v in verification['selected_checks'])
    lo,_,hi=ci['ratios']['v2']['bootstrap_percentiles_025_50_975']
    lines=['# V2：保留取樣與畫質的執行成本研究','',
        '**平均 10× 目標仍未達成。** 本輪已完成候選實作、淘汰實驗、選定版本及獨立場景複驗。',
        f'原十四場景的 V2 對逐像素基準加速 **{v2["aggregate_speedup"]:.2f}×**；同一次交錯量測比 V1 快 **{improvement:.3f}×**，GPU 耗時減少 **{100*reduction:.1f}%**。','',
        '[原尺寸切換比較](results/v2_final_1440/compare.html) · [原尺寸局部圖](results/v2_final_1440/native_crops.png) · '
        '[量測 JSON](results/v2_final_1440/metrics.json) · [實驗 CSV](results/v2_experiments.csv)','',
        '## 同一次量測的比較','',
        'RTX 4070，2560×1440，十四場景，每版本九次交錯 GPU 計時。每張重建探針、查表與細化遮罩；沒有跨幀重用影像，也沒有使用參考圖當輸入。','',
        '| 版本 | 平均 GPU ms | 總耗時加速 | 最慢場景加速 | 最差 RMSE | 最低 SSIM |',
        '|---|---:|---:|---:|---:|---:|']
    for key,label in (('full','完整基準'),('anchor','指定品質錨點'),('v1','V1'),('v2','V2')):
        a=s[key];lines.append(f'| {label} | {a["mean_ms"]:.3f} | {a["aggregate_speedup"]:.2f}× | {a["min_scene_speedup"]:.2f}× | {a["worst_rmse"]:.6f} | {a["min_ssim"]:.6f} |')
    lines += ['',f'十二個體積雲場景排除純天空後，加速 **{data["cloud_summary"]["v2"]["aggregate_speedup"]:.2f}×**。主數字以各場景基準中位數總和除以 V2 中位數總和計算，不採場景倍數的算術平均。',
        f'樣本 bootstrap 95% 區間為 {lo:.2f}–{hi:.2f}×；V1/V2 比值的區間為 {paired_ci[0]:.3f}–{paired_ci[2]:.3f}×。這些區間不含不同硬體、驅動、長期系統負載或不同雲模型的變動。',
        f'十四場景中，V2 中位數慢於 V1 的場景：{", ".join(failed_v1) or "無"}。純天空及小解析度採相同路徑，微小計時差異不代表演算法差異。',
        '', '![每場景耗時](results/v2_final_1440/per_scene.png)','',
        '## 選定 V2 的改動','',
        'V1 先算完整背景天空，再算雲，最後計算 `cloud.rgb + sky.rgb * cloud.transmittance`。V2 先算雲，僅在透光率大於零時算背景天空。完全不透明的雲後方天空不影響結果，可以省略。此處使用嚴格的 `> 0.0`，沒有新增近似的不透明門檻。',
        '粗層／細層解析度、T=0.035 梯度判斷、HDR 雙線性內插、密度場、陰影步數及積分位置都沿用 V1。純天空沿用 V1 完整解析度路徑；小於 800×450 像素仍直接回退原單 pass。',
        '改變計算順序也可能改變編譯器暫存器配置；本次沒有取得硬體計數器，不能把全部收益歸因於省略天空積分或宣稱已證明暫存器瓶頸。',
        '實作：[single_frame_v2.py](program/single_frame_v2.py)、[source transformations](program/v2_sources.py)。V1 僅新增可覆寫的 `render_levels` 呼叫點，預設執行內容不變。實驗版本集中在 [v2.py](program/v2.py)，選定版本不建立其額外 shader、佇列或 transport buffer。','',
        '## 畫質與功能驗證','',
        '品質錨點仍是 `confirm_1440/comparison.png` 的原 L3/N2/T=.04 管線。逐場景重新計算其 RMSE、SSIM 和誤差 >.03 的像素比例；V2 三項均不劣於錨點（容差 1e-6）。這些數值及影像比較不構成人眼盲測或所有場景的無損保證。',
        f'選定 V2 在功能驗證中的 {len(verification["selected_checks"])} 組影像，與 V1 的 HDR 結果有 **{identical} 組逐位元相同**；涵蓋原十四場景 1440p、三組奇數尺寸與三組小圖。另驗證首次影像、不依賴 warmup、A→B→A 重渲染及 OpenGL error。',
        '保留的 GPU 佇列實驗也通過 801×451 奇數尺寸、空／混合／全部細化、32/64/128 工作組、佇列像素無遺漏且無重複的檢查。這只證明功能正確，沒有使該路線通過效能要求。',
        '[功能驗證 JSON](results/verify_v2/verification.json)；所有原始碼雜湊與當時快照保留在各結果資料夾。','',
        '## 新場景及解析度複驗','',
        f'選定參數後，使用新 seed={held["metadata"]["heldout_seed"]} 產生十二組相機平移、高度、視角、時間與太陽方向，每版本七次交錯計時。這組配置未用於本輪候選篩選。總耗時加速 **{h["v2"]["aggregate_speedup"]:.2f}×**，最低 **{h["v2"]["min_scene_speedup"]:.2f}×**，品質通過 12/12。它仍來自同一程序雲模型。',
        '[新場景量測](results/v2_heldout_1440/metrics.json) · [新場景影像](results/v2_heldout_1440/compare.html)',
        f'800×450 原十四場景複驗加速 **{m["v2"]["aggregate_speedup"]:.2f}×**，最低 **{m["v2"]["min_scene_speedup"]:.2f}×**，品質通過 14/14。[資料](results/v2_confirm_800/metrics.json)','',
        '## 淘汰的候選','',
        '以下均有相同回合的 V1／完整基準對照；四場景子集為 A-g、A-t、B-t、preset-front，不能把其平均數直接與十四場景比較。','',
        '| 候選 | 結果 |', '|---|---|',
        '| GPU 光線佇列 + indirect compute | 保留品質，但 1440p 整體約 1.28–1.39× 基準，V1 同輪 3.84×。密集場景比逐像素基準還慢。直接遮罩 compute 也慢，並非只有排隊成本。 |',
        '| Catmull–Rom 內插並限制 overshoot | 未取得通過錨點品質的更快組合；T=.06 的 A-t RMSE .003634、SSIM .99179，劣於錨點 .002653／.99500。 |',
        '| 陰影 chunk 的第一 octave 上下界 | 可少算部分第二 octave，四場景品質相同，但整體收益不足且部分場景更慢。 |',
        '| 將雲傳輸與大氣合成拆成兩個 pass | 保持品質，但同輪平均 17.06 ms，V1 16.63 ms；比選定 V2 慢。 |',
        '| 延後 4000 m 陰影屋頂錨點 | 光線沒有走到時可省略，但實測新增分支版本較慢；合併 octave 上下界仍不及選定版本的整體結果。 |',
        '', '## 尚未解決的部分','',
        f'相同十四場景要達到 10×，平均時間需不超過 {s["full"]["mean_ms"]/10:.2f} ms，還要再削減 {100*(1-s["full"]["mean_ms"]/10/v2["mean_ms"]):.1f}% 的 V2 成本。',
        '密集雲層仍需大量逐光線密度與陰影積分。V2 保住 V1 畫質並降低少量成本，沒有達到平均 10×；也不能保證任意場景、硬體和解析度都不降速。要縮短剩餘差距，需要進一步研究更便宜的雲密度／光照表示或有可驗證誤差控制的空間重用。這是後續研究方向，尚未實作成符合要求的技術。',
        '整段 GPU timer 包含前置計算、分類、重建與繪製，不含首次 shader 編譯、資源配置、讀回或 CPU 品質分析。已保存獨立單張結果；計時重複相同參數僅用於估計速度，不使用前張結果。','',
        '## 重現','', '```bash',
        'python 9_cascade_sky_rendering/program/benchmark.py --research --experiment v2-selected --width 2560 --height 1440 --frames 9 --output v2_final_1440 --temporal',
        'python 9_cascade_sky_rendering/program/benchmark.py --research --experiment v2-selected --heldout --heldout-seed 20261003 --width 2560 --height 1440 --frames 7 --output v2_heldout_1440',
        'python 9_cascade_sky_rendering/program/benchmark.py --research --experiment v2-selected --width 800 --height 450 --frames 7 --output v2_confirm_800',
        'python 9_cascade_sky_rendering/program/benchmark.py --verify-v2 --output verify_v2',
        'MPLCONFIGDIR=/tmp/cascade_sky_mpl python 9_cascade_sky_rendering/program/analyze_v2.py',
        '```','',
        '研究佇列可使用 `--experiment v2 --variant-keys anchor,v1,compute0,compute32,compute64 --visual-variant compute64`。其他候選名稱和完整設定在各結果 `metrics.json` 的 `variant_definitions`。','']
    (ROOT/'V2_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(v2=s['v2'],same_run_v1_speedup=improvement,new_heldout=h['v2'],hdr_identical=identical),indent=2))


if __name__=='__main__':
    main()
