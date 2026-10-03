"""Report measured progress against the 10x target, including failed approaches."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/'results'


def read(name):
    return json.loads((RESULTS/name/'metrics.json').read_text())


def uncertainty(data):
    rng=np.random.default_rng(918)
    n=4000
    ratios={}
    base=np.zeros(n)
    vals={k:np.zeros(n) for k in data['summary']}
    for row in data['records']:
        for key,v in row['variants'].items():
            a=np.asarray(v['samples_ms'])
            draws=rng.choice(a,(n,len(a)),replace=True)
            vals[key]+=np.median(draws,axis=1)
    for key,v in vals.items():
        q=np.percentile(vals['full']/v,[2.5,50,97.5])
        ratios[key]=dict(bootstrap_percentiles_025_50_975=q.tolist())
    return dict(repetitions=n,seed=918,method='resample GPU timing samples independently within each scene/variant, sum scene medians',ratios=ratios)


def plot_times(data,dest):
    names=[r['scene']['name'] for r in data['records']]
    fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(len(names))
    for i,(key,label,color) in enumerate([('full','Full-resolution reference','#626b76'),
                ('anchor','Original cascade, L3 / N2 / T=.04','#dc9635'),
                ('optimized','Single-image pipeline, N4 / T=.035','#247cbd')]):
        y=[r['variants'][key]['median_ms'] for r in data['records']]
        ax.bar(x+(i-1)*.25,y,.25,label=label,color=color)
    ax.set_xticks(x,names,rotation=40,ha='right')
    ax.set_ylabel('Whole-pipeline GPU median (ms)')
    ax.legend(fontsize=9)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(dest,dpi=160)
    plt.close(fig)


def crops(folder,names):
    side=384
    sheet=Image.new('RGB',(side*4,(side+24)*len(names)),(18,18,22))
    draw=ImageDraw.Draw(sheet)
    coords={}
    for row,name in enumerate(names):
        images=[Image.open(folder/(name+'_'+suffix+'.png')).convert('RGB')
                for suffix in ('full','anchor','cascade')]
        a,b=np.asarray(images[0],dtype=float)/255,np.asarray(images[2],dtype=float)/255
        mse=np.mean((a-b)**2,axis=2)
        score=uniform_filter(mse,size=side,mode='constant')
        half=side//2
        inner=score[half:-half,half:-half]
        y,x=np.unravel_index(np.argmax(inner),inner.shape)
        coords[name]=dict(x=int(x),y=int(y),width=side,height=side)
        error=Image.fromarray(np.uint8(np.clip(np.abs(a-b)*8,0,1)*255+.5))
        for col,(im,title) in enumerate(zip(images+[error],('Full','Original cascade','Selected','Selected error x8'))):
            box=(int(x),int(y),int(x+side),int(y+side))
            sheet.paste(im.crop(box),(col*side,row*(side+24)+24))
            draw.text((col*side+4,row*(side+24)+5),name+' | '+title,fill='white')
    sheet.save(folder/'native_crops.png')
    (folder/'crop_coordinates.json').write_text(json.dumps(coords,indent=2)+'\n')


def viewer(folder,data):
    names=[r['scene']['name'] for r in data['records']]
    html='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>單張體積雲原尺寸比較</title><style>
body{background:#181a20;color:#eee;font:16px system-ui;margin:20px}button,select{font:inherit;margin:4px;padding:6px}
#viewport{height:76vh;overflow:auto;border:1px solid #777}#picture{display:block;max-width:none}
.active{background:#90c6f2}p{max-width:1000px}#label{font-weight:bold}
</style><h1>單張體積雲原尺寸比較</h1>
<p>影像預設以一個影像像素對一個 CSS 像素顯示。可捲動檢查細節；瀏覽器與螢幕縮放仍會影響實際顯示。
按 1、2、3 切換完整基準、原階層與本次版本。肉眼判讀不能由縮小總覽或 SSIM 取代。</p>
<select id="scene"></select><button data-mode="full">1 完整基準</button><button data-mode="anchor">2 原階層</button>
<button data-mode="cascade">3 本次版本</button><button id="blink">完整基準／本次版本交替</button>
<button id="fit">切換符合寬度／原尺寸</button><p id="label"></p><div id="viewport"><img id="picture"></div>
<script>const names=SCENE_NAMES;let mode='cascade',timer=null,fit=false;
const scene=document.getElementById('scene'),picture=document.getElementById('picture');
names.forEach(n=>scene.add(new Option(n,n)));
function show(){picture.src=scene.value+'_'+mode+'.png';document.getElementById('label').textContent=scene.value+' — '+mode;
document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));}
scene.onchange=show;document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;show()});
document.onkeydown=e=>{if(['1','2','3'].includes(e.key)){mode=['full','anchor','cascade'][Number(e.key)-1];show()}};
document.getElementById('blink').onclick=()=>{if(timer){clearInterval(timer);timer=null;}else timer=setInterval(()=>{mode=mode==='full'?'cascade':'full';show()},700)};
document.getElementById('fit').onclick=()=>{fit=!fit;picture.style.maxWidth=fit?'100%':'none';};show();</script></html>'''
    (folder/'compare.html').write_text(html.replace('SCENE_NAMES',json.dumps(names)),encoding='utf-8')


def main():
    folder=RESULTS/'single_frame_final_1440'
    d=read(folder.name)
    if len(d['records'])!=14:
        raise RuntimeError('Wait for the complete 14-scene confirmation')
    s=d['summary']; c=d['cloud_summary']
    plot_times(d,folder/'per_scene.png')
    crops(folder,['A-t','B-t','B-h','preset-front','postset-side'])
    viewer(folder,d)
    ci=uncertainty(d)
    (folder/'bootstrap_summary.json').write_text(json.dumps(ci,indent=2)+'\n')
    heldout=None
    p=RESULTS/'single_frame_heldout_1440'/'metrics.json'
    if p.exists():
        heldout=json.loads(p.read_text())
        if len(heldout['records'])!=12:
            raise RuntimeError('Wait for all 12 held-out scenes')
        plot_times(heldout,p.parent/'per_scene.png')
        viewer(p.parent,heldout)
    aggressive=read('single_frame_1440')['summary']['aggressive']
    rows=[]
    for path in sorted(RESULTS.glob('*/metrics.json')):
        x=json.loads(path.read_text())
        if 'research' not in x.get('metadata',{}).get('timing','') and not any(
                k in x['summary'] for k in ('shared-full','lookup-full','hash-full','roof1024-full',
                    'candidates-full','candidates-N4','chunks-full','bounded-full','lighting3-full',
                    'separable4-T0.025','transport-0.02','split-0.05','optimized')):
            continue
        for key,v in x['summary'].items():
            if key=='full':continue
            rows.append([path.parent.name,len(x['records']),key,v['aggregate_speedup'],
                         v['worst_rmse'],v['min_ssim'],v['quality_gate_pass']])
    with (RESULTS/'single_frame_experiments.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['dataset','scene_count','variant','aggregate_speedup','worst_rmse','min_ssim','legacy_loose_gate']);writer.writerows(rows)
    optimized=s['optimized'];anchor=s['anchor']
    lower,_,upper=ci['ratios']['optimized']['bootstrap_percentiles_025_50_975']
    lines=[
        '# 單張體積雲：10× 目標的後續研究',
        '',
        '**狀態：尚未達成平均 10× 的完整要求。** 本次選定版本保留單張、無歷史影像的管線，'
        f'在原十四場景實測總耗時加速 **{optimized["aggregate_speedup"]:.2f}×**，'
        f'最慢場景仍有 **{optimized["min_scene_speedup"]:.2f}×**。不能把這當成所有可能場景都不降速的保證。',
        '',
        '[原尺寸切換比較](results/single_frame_final_1440/compare.html) · '
        '[原尺寸局部圖](results/single_frame_final_1440/native_crops.png) · '
        '[完整量測 JSON](results/single_frame_final_1440/metrics.json) · '
        '[所有實驗 CSV](results/single_frame_experiments.csv)',
        '',
        '## 對指定品質與速度的檢查',
        '',
        '以 `confirm_1440/comparison.png` 對應的 L3／N2／T=0.04 為品質錨點（anchor），'
        '在每個新測量場景重新渲染；完整逐像素 PSMS Next_2 仍是誤差與效能的共同基準。'
        '所有相依前置計算每張影像重建，包含在整條 GPU timer 內。沒有讀取預存參考圖當渲染輸入。',
        '',f'測試環境：{d["metadata"]["gpu"]}，2560×1440，每個場景／版本九次交錯計時。',
        '',
        '| 管線 | 平均 GPU ms | 總耗時加速 | 最慢場景加速 | 最差 RMSE | 最低 SSIM |',
        '|---|---:|---:|---:|---:|---:|']
    for key,label in [('full','完整基準'),('anchor','指定的原階層'),('optimized','本次選定版本')]:
        v=s[key];lines.append(f'| {label} | {v["mean_ms"]:.3f} | {v["aggregate_speedup"]:.2f}× | {v["min_scene_speedup"]:.2f}× | {v["worst_rmse"]:.6f} | {v["min_ssim"]:.6f} |')
    failures=[r['scene']['name'] for r in d['records'] if not r['variants']['optimized']['quality_vs_anchor']['pass']]
    lines += ['',f'逐場景要求 RMSE 不增加、SSIM 不降低、誤差 >0.03 的像素比例不增加（浮點容差各 1e-6）。'
              f'本輪通過 {14-len(failures)}/14；未通過：{", ".join(failures) or "無"}。',
              '', '這是比原先 RMSE≤0.005／SSIM≥0.99 更嚴格的相對檢查，但仍不是「任何人肉眼完全看不出」的證明。'
              '總覽圖會縮小影像；應使用上述原尺寸切換器與局部圖判讀。',
              '',f'排除兩張純天空後，十二個體積雲場景的總耗時加速為 **{c["optimized"]["aggregate_speedup"]:.2f}×**。'
              f'十四場景倍數的算術平均為 {optimized["arithmetic_mean_scene_speedup"]:.2f}×；本報告使用「基準總時間／新方法總時間」作主要數字，避免便宜天空場景誇大平均。',
              '',f'九次交錯計時樣本的 bootstrap 區間為 {lower:.2f}–{upper:.2f}×（2.5–97.5 百分位）；'
              '它只反映本輪樣本的抽樣變動，不包含跨機器、驅動、場景或長時間系統負載的不確定性。',
              '', '![逐場景耗時](results/single_frame_final_1440/per_scene.png)',
              '', '## 最終選用的實作', '',
              '1. 兩個像素一次計算本張影像的雲環境光探針，取代每個光線重算。',
              '2. 球對稱大氣 diffuse gather 以高度／太陽天頂角的 512×256 查表重建；雲形事件參數以 64×32 表重用。',
              '3. 每個 500 m 方格建立候選雲頂事件 bitmask。對每個橢圓事件計算方格內高度上下界，剔除不可能超過最大下界的事件；保留事件仍使用原解析函式求值，沒有以平滑高度圖取代雲形。',
              '4. 雲只在 40 km fade 內有效。先排除不會進入積分的射線，讓雲頂候選表的 >120 km 邊界有足夠範圍；4000 m 光線屋頂錨點也包含在內。這個範圍證明依賴本快照的 profile 常數，改變雲模型時必須重查。',
              '5. 直接從 W/4×H/4 細化至 W×H（N=4、兩層），T=0.035，取消中間重算層。梯度分數只在上一層每個 footprint 算一次，再給最後一層共用。',
              '6. 純天空使用完整解析度，僅共用大氣 gather 表，保留程序星點與日月細節。小於 800×450 像素時直接使用原單 pass，避免固定多 pass 成本；不先跑前置步驟再回退。',
              '', '主要程式：[single_frame.py](program/single_frame.py)、[source transformations](program/optimized_sources.py)、'
              '[parent score](program/parent_score.frag)、[reconstruction](program/scored_reconstruct.frag)。'
              '實驗版本另留在 `fast.py`，沒有覆寫研究 0～3 或原始 `confirm_1440` 結果。這仍是獨立 OpenGL 研究管線，尚未整合進模擬器的 frame graph。',
              '', '## 沒有採用的路線', '',
              '| 嘗試 | 實測結論 |', '|---|---|',
              '| 曲率代替梯度 | 在密集雲層沒有更好的速度／品質組合。 |',
              '| 33M 整數雜訊表 | 記憶體讀取成本高，且部分編譯／浮點差異造成可量測圖像誤差。 |',
              '| 1024²／2048² 平滑雲頂圖 | 雖更快，插值誤差會改變適應性步進；俯視品質差於錨點。 |',
              '| 低頻密度格點三維快取 | 能保持主要輸出，仍比解析路徑慢。 |',
              '| 陰影 stride 3／4、粗陰影步數減半 | stride 3 的 A-t RMSE 已到 0.00636；不符合指定品質。 |',
              '| 水平／垂直分開細化 | 地面場景有收益，俯視細節誤差過高。 |',
              '| 保留密度、共用粗層照明 | 小門檻加速不足；大門檻改變明暗。拆成獨立 shade pass 也未解決。 |',
              '| 放寬影像梯度門檻 | 可換速度，但直接違反品質要求。 |',
              '',f'一個刻意保留的失敗對照（L3／N2／T=.12）在十四場景的算術平均達 '
              f'{aggressive["arithmetic_mean_scene_speedup"]:.2f}×，然而總耗時只有 {aggressive["aggregate_speedup"]:.2f}×，'
              f'最差 RMSE={aggressive["worst_rmse"]:.6f}、SSIM={aggressive["min_ssim"]:.5f}，'
              f'{100*aggressive["worst_bad_pixel_fraction"]:.2f}% 像素誤差 >0.03。這不是完成 10× 目標。',
              '', '## 獨立與功能驗證', '']
    if heldout:
        h=heldout['summary']['optimized']
        fails=[r['scene']['name'] for r in heldout['records'] if not r['variants']['optimized']['quality_vs_anchor']['pass']]
        lines += [f'固定 seed=90210，另取十二組未參與參數調整的相機平移、高度、視角、時間和太陽方向。'
                  f'總耗時加速 {h["aggregate_speedup"]:.2f}×，最慢場景 {h["min_scene_speedup"]:.2f}×；'
                  f'逐場景相對品質通過 {12-len(fails)}/12，未通過：{", ".join(fails) or "無"}。'
                  '這是同一程序雲模型的新觀測，不是不同雲模型的普遍驗證。',
                  '', '[獨立場景資料](results/single_frame_heldout_1440/metrics.json) · '
                  '[獨立場景影像](results/single_frame_heldout_1440/compare.html)', '']
    vpath=RESULTS/'verify_single_frame_final'/'verification.json'
    if vpath.exists():
        v=json.loads(vpath.read_text())
        lines += [f'GPU 功能驗證：801×451 奇數尺寸強制全渲染與 optimized full 逐位元相同；'
                  f'CPU 雙線性內插對照的最大 display 誤差 '
                  f'{max(r["force_interpolate_max_display"] for r in v["cases"]):.2g}。'
                  f'三組相機／時間各 {v["roof_field_samples_per_case"]:,} 個雲頂位置與原解析式比較，最大高度誤差 '
                  f'{max(v["roof_field_max_error_metres"]):g} m。A→B→A 重渲染逐位元相同，小圖回退也與原版逐位元相同。',
                  '', '驗證另修正了首次建立 framebuffer 覆蓋當前紋理單元的問題：新表面配置現在保存並還原綁定。'
                  '大氣 gather 的 LUT 近似與 shader 編譯差異仍有殘差；遠座標診斷的全解析度 RMSE 最大約 0.000248，沒有宣稱所有計算 bit-exact。',
                  '', '[功能驗證資料](results/verify_single_frame_final/verification.json)', '']
    for name,label in [('single_frame_800','800×450'),('single_frame_small','320×180')]:
        p=RESULTS/name/'metrics.json'
        if p.exists():
            z=json.loads(p.read_text()); q=z['summary']['optimized']
            note=('小圖使用相同原始繪製路徑，時間仍會有量測雜訊。' if name=='single_frame_small'
                  else '全部場景均加速且通過相對品質檢查。')
            lines += [f'{label} 複驗：{len(z["records"])} 場景，總耗時加速 {q["aggregate_speedup"]:.2f}×，'
                      f'最低場景比 {q["min_scene_speedup"]:.2f}×。{note}'
                      f' [資料](results/{name}/metrics.json)', '']
    lines += ['## 限制與研究判斷','',
              '主要限制仍是密集雲層的逐光線密度與陰影工作。當大部分區域必須細化，空間採樣省不了足夠工作；'
              '其餘已測試的捷徑不是變慢，就是消耗超過允許的品質誤差。這些實驗不能證明 10× 不可能，但目前不能宣稱已實現。',
              '', '初始粗採樣看不到的細雲仍可能被漏掉；顏色梯度不是誤差上界。1440p 與 800×450 測試均加速，也不能推出所有視角、硬體、解析度都沒有降速。'
              '畫質參考是原 PSMS 近似 shader，不是收斂的物理體積散射；「肉眼不可分辨」尚無盲測證據。',
              '', 'GL_TIME_ELAPSED 涵蓋全部前置步驟、分類、重建、stencil 與 shade，但不含首次 shader 編譯、資源配置、讀回與 CPU 指標。'
              'Python／driver 提交空檔可能進入整段 GPU timer；逐 pass 計時條件不同，其加總不等於整段，不能用來替換主結果。沒有跨幀保留渲染結果。',
              '', '## 重現', '', '```bash',
              'python 9_cascade_sky_rendering/program/benchmark.py --verify-selected --output verify_single_frame_final',
              'python 9_cascade_sky_rendering/program/benchmark.py --research --experiment selected --variant-keys anchor,optimized --threshold .035 --width 2560 --height 1440 --frames 9 --output single_frame_final_1440 --profile --temporal',
              'python 9_cascade_sky_rendering/program/benchmark.py --research --experiment selected --variant-keys anchor,optimized --threshold .035 --heldout --width 2560 --height 1440 --frames 7 --output single_frame_heldout_1440',
              'MPLCONFIGDIR=/tmp/cascade_sky_mpl python 9_cascade_sky_rendering/program/analyze_single_frame.py',
              '```', '', '每個資料夾保留當時的 source snapshot、SHA-256、GPU、場景參數與每次計時，原研究資料未覆蓋。',
              '', '## 相關方法', '',
              '共用大氣查表的方向與 [Hillaire 2020](https://sebh.github.io/publications/egsr2020.pdf) 的方法相關；'
              '本實作保留 PSMS 的散射近似，沒有宣稱移植其完整模型。'
              '[Nubis, Evolved](https://www.guerrilla-games.com/read/nubis-evolved) 也說明單張高品質雲需要體積表示與採樣成本的整體設計。'
              '階層式體積採樣本身可追溯至 [Levoy 1990](https://graphics.stanford.edu/papers/Levoy-hpscans/adaptive-viscomp90/INDEX.HTM)，此處不主張基本思想的新穎性。', '']
    (ROOT/'SINGLE_FRAME_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote SINGLE_FRAME_REPORT.md and visual artifacts')


if __name__=='__main__':main()
