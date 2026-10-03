"""Build the research report/plots directly from recorded measurements."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def table(data):
    lines = ['| 版本 | 平均 ms | 總耗時加速比 | 最差 RMSE | 最低 SSIM | 品質門檻 |',
             '|---|---:|---:|---:|---:|---|']
    for name,s in data['summary'].items():
        lines.append('| %s | %.3f | %.2f× | %.6f | %.5f | %s |' %
            (name,s['mean_ms'],s['aggregate_speedup'],s['worst_rmse'],s['min_ssim'],
             '通過' if s['quality_gate_pass'] else '未通過'))
    return '\n'.join(lines)


def plot_quality(datasets, out):
    fig, axes = plt.subplots(1,len(datasets),figsize=(7*len(datasets),5),squeeze=False)
    for ax,(name,data) in zip(axes[0],datasets.items()):
        if name.startswith('800'):
            offsets = {'L3-T0.005':(-8,28),'L4-T0.005':(-8,-20),
                       'L3-T0.01':(2,-25),'L4-T0.01':(0,-40),
                       'L3-T0.02':(15,20),'L4-T0.02':(20,-5),
                       'L2-T0.01':(-30,40),'L2-T0.02':(0,25)}
        else:
            offsets = {'L3-T0.04':(-45,-20),'L4-T0.04':(10,20),
                       'L3-T0.08':(8,13),'L3-T0.04-R0':(8,-12)}
        for key,s in data['summary'].items():
            if key=='full':
                continue
            color = '#176f65' if s['quality_gate_pass'] else '#bd553e'
            ax.scatter(s['worst_rmse'],s['aggregate_speedup'],c=color,s=50)
            offset = offsets.get(key,(4,4))
            if key.startswith('uniform-'):
                offset = (-5,-15)
            ax.annotate(key,(s['worst_rmse'],s['aggregate_speedup']),xytext=offset,
                        textcoords='offset points',fontsize=7,
                        ha='right' if key.startswith('uniform-') else 'left',
                        arrowprops=dict(arrowstyle='-',color='#888888',lw=.5) if key in offsets else None)
        ax.axvline(.005,c='gray',ls='--',lw=1,label='RMSE gate (SSIM/bad pixels also required)')
        ax.axhline(1,c='gray',ls=':',lw=1)
        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Worst scene display RGB RMSE (lower is better)')
        ax.set_ylabel('Aggregate speedup vs full shading')
        ax.set_title(name+' | green: all quality gates pass')
        ax.grid(alpha=.2)
        ax.text(.98,.025,'Dashed: RMSE gate; dotted: 1x baseline',transform=ax.transAxes,
                ha='right',fontsize=7,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    fig.tight_layout()
    fig.savefig(out/'quality_performance.png',dpi=160)
    plt.close(fig)


def scene_plot(data,key,out):
    rows = data['records']
    names = [r['scene']['name'] for r in rows]
    full = [r['variants']['full']['median_ms'] for r in rows]
    cascade = [r['variants'][key]['median_ms'] for r in rows]
    fig,ax = plt.subplots(figsize=(11,6))
    y = np.arange(len(rows))
    ax.barh(y-.18,full,.36,label='Full-resolution baseline',color='#767e89')
    ax.barh(y+.18,cascade,.36,label=key,color='#176f65')
    ax.set_yticks(y,names)
    ax.invert_yaxis()
    ax.set_xlabel('Entire GPU pipeline, median ms (lower is better)')
    ax.set_title('2560 x 1440 | RTX 4070 | per-scene results')
    ax.legend()
    ax.grid(axis='x',alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'per_scene_1440.png',dpi=160)
    plt.close(fig)


def worst_crops(directory,out):
    # Full-size crop centered on the largest mean error window, not hand-selected.
    scenes = ['A-t','B-g','B-h','postset-side']
    size = 384
    sheet = Image.new('RGB',(size*3,(size+25)*len(scenes)),(20,20,23))
    draw = ImageDraw.Draw(sheet)
    coords = {}
    for i,name in enumerate(scenes):
        a = Image.open(directory/(name+'_full.png')).convert('RGB')
        b = Image.open(directory/(name+'_cascade.png')).convert('RGB')
        aa,bb = np.asarray(a,dtype=np.float32)/255,np.asarray(b,dtype=np.float32)/255
        error = np.mean(np.abs(aa-bb),axis=2)
        best = (-1,0,0)
        for y in range(0,a.height-size+1,48):
            for x in range(0,a.width-size+1,48):
                value = float(error[y:y+size,x:x+size].mean())
                if value>best[0]:
                    best = (value,x,y)
        _,x,y = best
        coords[name] = [x,y,size,size]
        e = Image.fromarray(np.uint8(np.clip(np.abs(aa-bb)*8,0,1)*255+.5))
        for j,(title,img) in enumerate([('Full',a),('Cascade',b),('Abs difference x8',e)]):
            sheet.paste(img.crop((x,y,x+size,y+size)),(j*size,i*(size+25)+25))
            draw.text((j*size+4,i*(size+25)+6),name+' | '+title,fill=(235,235,235))
    sheet.save(out/'detail_crops_1440.png')
    (out/'crop_coordinates.json').write_text(json.dumps(coords,indent=2)+'\n')


def main():
    out = ROOT/'results'
    d800 = json.loads((out/'sweep_800'/'metrics.json').read_text())
    d1440 = json.loads((out/'confirm_1440'/'metrics.json').read_text())
    if len(d800['records'])!=14 or len(d1440['records'])!=14:
        raise RuntimeError('Wait for both 14-scene benchmarks to complete')
    datasets = {'800 x 450':d800,'2560 x 1440':d1440}
    plot_quality(datasets,out)
    key = 'L3-T0.04'
    scene_plot(d1440,key,out)
    worst_crops(out/'confirm_1440',out)
    with (out/'summary.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['resolution','variant','mean_ms','aggregate_speedup','worst_rmse','min_ssim','quality_gate'])
        for resolution,data in datasets.items():
            for name,s in data['summary'].items():
                writer.writerow([resolution,name,s['mean_ms'],s['aggregate_speedup'],s['worst_rmse'],s['min_ssim'],s['quality_gate_pass']])
    selected = d1440['summary'][key]
    baseline = d1440['summary']['full']
    scenes = []
    profiles = []
    temporal = []
    for record in d1440['records']:
        name = record['scene']['name']
        full = record['variants']['full']
        v = record['variants'][key]
        scenes.append('| %s | %.3f | %.3f | %.2f× | %.6f | %.5f | %.1f%% |' %
            (name,full['median_ms'],v['median_ms'],v['speedup'],v['quality']['rmse'],
             v['quality']['ssim_luma'],100*v['work']['full_resolution_equivalent_shading']))
        p = v.get('draw_only_profile_ms',{})
        if p:
            shading = sum(t for stage,t in p.items() if stage.endswith('-shade'))
            classify = sum(t for stage,t in p.items() if 'classify' in stage)
            stencil = sum(t for stage,t in p.items() if 'stencil' in stage)
            profiles.append('| %s | %.3f | %.3f | %.3f | %.3f | %.3f |' %
                (name,v['median_ms'],shading,classify,stencil,sum(p.values())))
        temporal.append((name,v['temporal']['time_error_delta_rmse'],v['temporal']['camera_error_delta_rmse']))
    worst_time = max(temporal,key=lambda x:x[1])
    worst_camera = max(temporal,key=lambda x:x[2])
    verification = json.loads((out/'verification'/'verification.json').read_text())
    report = '''# Cascade Sky Rendering — 實測研究結果

## 本次結論

RTX 4070、2560×1440、14 場景，三層 `N=2,T=0.04,radius=1`：

- 每場景 median 的平均由 **%.3f ms 降到 %.3f ms**，總耗時比 **%.2f×**，減少 **%.1f%%**。
- 最差顯示 RGB RMSE **%.6f**，最低 luminance SSIM **%.5f**；本次品質門檻 **%s**。
- 最慢場景的相對速度為 **%.2f×**；此策略不是所有視角都加速。

以上是整個階層 GPU 管線的量測，包含分類、內插、stencil 及各層完整 shader。參考是同一 PSMS shader 的逐像素全解析度結果，不是高精度物理 ground truth。

## 設備與量測設定

- GPU：%s；OpenGL：%s。
- 800×450：每變體每場景 7 次；2560×1440：9 次。每場景先預熱，再輪替／反轉變體順序。
- 12 個既有雲場景＋2 個純大氣場景；固定 cloudTime，日落矩陣含 +4° 和 −1° 太陽仰角。
- 每次結果存原始 GPU timer 樣本、median、p95、完整參數及源碼 SHA-256。兩個解析度各自在同一輪內比較，不用跨輪絕對時間算加速。
- 操作性品質門檻：所有場景 RMSE ≤0.005、SSIM ≥0.99、任一 RGB 通道誤差 >0.03 的像素比例 ≤1%%。門檻不代表視覺無損。
- 總加速比 = 各場景 baseline median 加總 / 各場景 variant median 加總，不是各場景加速比的平均。

## 800×450 參數掃描

%s

## 2560×1440 參數比較

%s

`L3` 包含初始層，共三層；`T` 是顯示 RGB 鄰接差門檻；`R0` 不含一像素 halo，其餘使用 `radius=1`。`uniform-*` 強制只內插，是流程消融對照，仍含同一管線的遮罩與空 shade draw 成本。

![品質與效能](results/quality_performance.png)

## 三層 T=0.04：逐場景結果

| 場景 | 原版 ms | 階層 ms | 加速比 | RMSE | SSIM | 相當全解析度 shade 量 |
|---|---:|---:|---:|---:|---:|---:|
%s

最後一欄包含所有層的 shade 像素，不是只有最終層遮罩。它沒有把不同光線的積分成本和 SIMD 利用率視為相同，因此不能直接倒數當成加速比。

![逐場景時間](results/per_scene_1440.png)

## GPU pass 成本診斷

| 場景 | 整體管線 ms | shade draws ms | 分類＋內插 draws ms | stencil draws ms | draw 合計 ms |
|---|---:|---:|---:|---:|---:|
%s

Draw 計時在另一組診斷重複中量測，並取各 pass median；與整體 timer 不是同一個巢狀 interval，不能把差值當成精確的單一成本。它也不含 pass 間切換、clear、Python/driver 提交空檔等。主結論始終使用完整管線 timer。

## 圖像與動態診斷

- [1440p 全場景比較](results/confirm_1440/comparison.png)：原版、階層、8 倍差異、最終層遮罩；白色遮罩表示重新 shade。
- [800p 掃描比較圖](results/sweep_800/comparison.png)：該輪圖像使用 T=0.01，與正式 1440p 的 T=0.04 不同，詳見各輪 metadata。
- 下圖以 PNG 顯示色彩差異的局部平均最大區域選取 384×384 原尺寸 crop，沒有放大內插。誤差 x8 為從輸出 PNG 重算的可視化，正式數值使用未量化浮點資料。

![最差區域原尺寸比較](results/detail_crops_1440.png)

對 T=0.04 的誤差變化診斷，雲時間前進 1/60 秒時，最差 RMSE 為 **%.6f**（%s）；相機平移 10 公尺時為 **%.6f**（%s）。這是兩幀之間重建誤差的變化，尚未做長時間影片或 temporal reconstruction。

## 正確性與反例

- N=2 與 N=3、奇數解析度 161×91，強制全部細化都與本研究全解析度 wrapper 逐像素一致，最大 HDR 差為 0。
- 強制全部內插，與獨立 CPU 雙線性內插的最大 HDR 差：N=2 為 %.3g，N=3 為 %.3g。
- 原 GLSL 330 wrapper 對照本研究 GLSL 430／uniform wrapper：顯示 RMSE %.3g。兩種編譯形式有小量浮點差異；所有主比較則使用完全相同的研究 wrapper。
- 單像素亮點反例：初始層完全漏採樣後，後續層梯度為零，亮點持續遺失。這個預期失敗被保留在 [verification.json](results/verification/verification.json)，不包裝成可由梯度保證收斂。

## 解讀與下一步

1. 初始假設在低變化天空與較平滑的雲視角成立，高解析度更能攤薄多 pass 成本。密集俯視雲紋則容易反覆細化，不能僅以天空占比預測收益。
2. 更高門檻或更小 halo 會減少重算，但要看所有場景的品質；高對比細碎雲紋是重要壓力測試。表格保留未通過的設定。
3. 未來可加入 coarse 細化密度驅動的全解析度 fallback，降低密集畫面的負收益；本版尚未實作，不能宣稱已避免變慢。
4. 未被初始層看見的細節需要額外訊息，例如保底探索、幾何／深度／光學厚度遮罩。單純調低顏色梯度門檻不能解決零梯度的漏採樣。
5. 下一個部署階段是引擎 frame graph 整合及更長的相機／雲動畫序列；這些尚未量測。亦尚未測 RGBA16F、時域重投影或多 GPU。

完整算法、公式與重現指令見 [README](README.md)。
''' % (baseline['mean_ms'],selected['mean_ms'],selected['aggregate_speedup'],
       100*(1-selected['mean_ms']/baseline['mean_ms']),selected['worst_rmse'],selected['min_ssim'],
       '通過' if selected['quality_gate_pass'] else '未通過',selected['min_scene_speedup'],
       d1440['metadata']['gpu'],d1440['metadata']['opengl'],table(d800),table(d1440),
       '\n'.join(scenes),'\n'.join(profiles),worst_time[1],worst_time[0],worst_camera[2],worst_camera[0],
       verification['force_none_N2_max_abs_hdr'],verification['force_none_N3_max_abs_hdr'],
       verification['original_glsl330_vs_wrapper_display_rmse'])
    (ROOT/'REPORT.md').write_text(report)
    extra = out/'factor3_check'/'metrics.json'
    if extra.exists():
        d3 = json.loads(extra.read_text())
        appendix = '\n## N=3 補測\n\n'
        appendix += ('810×459、三層、T=0.04、radius=1，使用自訂 CLI 參數，另跑 A-g、B-t、sky-day 三場景。'
                     '此輪用來驗證倍率可調，不與 1440p 或 800p 的絕對時間交叉相除。\n\n'
                     '| 場景 | 原版 ms | 階層 ms | 加速比 | RMSE |\n|---|---:|---:|---:|---:|\n')
        for r in d3['records']:
            b,v = r['variants']['full'],r['variants']['custom']
            appendix += '| %s | %.3f | %.3f | %.2f× | %.6f |\n' % (
                r['scene']['name'],b['median_ms'],v['median_ms'],v['speedup'],v['quality']['rmse'])
        appendix += '\n[原始資料](results/factor3_check/metrics.json)\n'
        with (ROOT/'REPORT.md').open('a') as f:
            f.write(appendix)
    print('Wrote REPORT.md, results/summary.csv and figures.')


if __name__=='__main__':
    main()
