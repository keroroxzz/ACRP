# 單張體積雲：10× 目標的後續研究

**狀態：尚未達成平均 10× 的完整要求。** 本次選定版本保留單張、無歷史影像的管線，在原十四場景實測總耗時加速 **5.11×**，最慢場景仍有 **2.46×**。不能把這當成所有可能場景都不降速的保證。

[原尺寸切換比較](results/single_frame_final_1440/compare.html) · [原尺寸局部圖](results/single_frame_final_1440/native_crops.png) · [完整量測 JSON](results/single_frame_final_1440/metrics.json) · [所有實驗 CSV](results/single_frame_experiments.csv)

## 對指定品質與速度的檢查

以 `confirm_1440/comparison.png` 對應的 L3／N2／T=0.04 為品質錨點（anchor），在每個新測量場景重新渲染；完整逐像素 PSMS Next_2 仍是誤差與效能的共同基準。所有相依前置計算每張影像重建，包含在整條 GPU timer 內。沒有讀取預存參考圖當渲染輸入。

測試環境：NVIDIA GeForce RTX 4070/PCIe/SSE2，2560×1440，每個場景／版本九次交錯計時。

| 管線 | 平均 GPU ms | 總耗時加速 | 最慢場景加速 | 最差 RMSE | 最低 SSIM |
|---|---:|---:|---:|---:|---:|
| 完整基準 | 67.190 | 1.00× | 1.00× | 0.000000 | 1.000000 |
| 指定的原階層 | 24.363 | 2.76× | 0.89× | 0.002653 | 0.995004 |
| 本次選定版本 | 13.143 | 5.11× | 2.46× | 0.001783 | 0.997859 |

逐場景要求 RMSE 不增加、SSIM 不降低、誤差 >0.03 的像素比例不增加（浮點容差各 1e-6）。本輪通過 14/14；未通過：無。

這是比原先 RMSE≤0.005／SSIM≥0.99 更嚴格的相對檢查，但仍不是「任何人肉眼完全看不出」的證明。總覽圖會縮小影像；應使用上述原尺寸切換器與局部圖判讀。

排除兩張純天空後，十二個體積雲場景的總耗時加速為 **5.17×**。十四場景倍數的算術平均為 5.59×；本報告使用「基準總時間／新方法總時間」作主要數字，避免便宜天空場景誇大平均。

九次交錯計時樣本的 bootstrap 區間為 5.09–5.13×（2.5–97.5 百分位）；它只反映本輪樣本的抽樣變動，不包含跨機器、驅動、場景或長時間系統負載的不確定性。

![逐場景耗時](results/single_frame_final_1440/per_scene.png)

## 最終選用的實作

1. 兩個像素一次計算本張影像的雲環境光探針，取代每個光線重算。
2. 球對稱大氣 diffuse gather 以高度／太陽天頂角的 512×256 查表重建；雲形事件參數以 64×32 表重用。
3. 每個 500 m 方格建立候選雲頂事件 bitmask。對每個橢圓事件計算方格內高度上下界，剔除不可能超過最大下界的事件；保留事件仍使用原解析函式求值，沒有以平滑高度圖取代雲形。
4. 雲只在 40 km fade 內有效。先排除不會進入積分的射線，讓雲頂候選表的 >120 km 邊界有足夠範圍；4000 m 光線屋頂錨點也包含在內。這個範圍證明依賴本快照的 profile 常數，改變雲模型時必須重查。
5. 直接從 W/4×H/4 細化至 W×H（N=4、兩層），T=0.035，取消中間重算層。梯度分數只在上一層每個 footprint 算一次，再給最後一層共用。
6. 純天空使用完整解析度，僅共用大氣 gather 表，保留程序星點與日月細節。小於 800×450 像素時直接使用原單 pass，避免固定多 pass 成本；不先跑前置步驟再回退。

主要程式：[single_frame.py](program/single_frame.py)、[source transformations](program/optimized_sources.py)、[parent score](program/parent_score.frag)、[reconstruction](program/scored_reconstruct.frag)。實驗版本另留在 `fast.py`，沒有覆寫研究 0～3 或原始 `confirm_1440` 結果。這仍是獨立 OpenGL 研究管線，尚未整合進模擬器的 frame graph。

## 沒有採用的路線

| 嘗試 | 實測結論 |
|---|---|
| 曲率代替梯度 | 在密集雲層沒有更好的速度／品質組合。 |
| 33M 整數雜訊表 | 記憶體讀取成本高，且部分編譯／浮點差異造成可量測圖像誤差。 |
| 1024²／2048² 平滑雲頂圖 | 雖更快，插值誤差會改變適應性步進；俯視品質差於錨點。 |
| 低頻密度格點三維快取 | 能保持主要輸出，仍比解析路徑慢。 |
| 陰影 stride 3／4、粗陰影步數減半 | stride 3 的 A-t RMSE 已到 0.00636；不符合指定品質。 |
| 水平／垂直分開細化 | 地面場景有收益，俯視細節誤差過高。 |
| 保留密度、共用粗層照明 | 小門檻加速不足；大門檻改變明暗。拆成獨立 shade pass 也未解決。 |
| 放寬影像梯度門檻 | 可換速度，但直接違反品質要求。 |

一個刻意保留的失敗對照（L3／N2／T=.12）在十四場景的算術平均達 10.22×，然而總耗時只有 8.69×，最差 RMSE=0.009148、SSIM=0.95612，5.11% 像素誤差 >0.03。這不是完成 10× 目標。

## 獨立與功能驗證

固定 seed=90210，另取十二組未參與參數調整的相機平移、高度、視角、時間和太陽方向。總耗時加速 5.04×，最慢場景 2.49×；逐場景相對品質通過 12/12，未通過：無。這是同一程序雲模型的新觀測，不是不同雲模型的普遍驗證。

[獨立場景資料](results/single_frame_heldout_1440/metrics.json) · [獨立場景影像](results/single_frame_heldout_1440/compare.html)

GPU 功能驗證：801×451 奇數尺寸強制全渲染與 optimized full 逐位元相同；CPU 雙線性內插對照的最大 display 誤差 1.1e-05。三組相機／時間各 252,003 個雲頂位置與原解析式比較，最大高度誤差 0 m。A→B→A 重渲染逐位元相同，小圖回退也與原版逐位元相同。

驗證另修正了首次建立 framebuffer 覆蓋當前紋理單元的問題：新表面配置現在保存並還原綁定。大氣 gather 的 LUT 近似與 shader 編譯差異仍有殘差；遠座標診斷的全解析度 RMSE 最大約 0.000248，沒有宣稱所有計算 bit-exact。

[功能驗證資料](results/verify_single_frame_final/verification.json)

800×450 複驗：14 場景，總耗時加速 2.28×，最低場景比 1.94×。全部場景均加速且通過相對品質檢查。 [資料](results/single_frame_800/metrics.json)

320×180 複驗：4 場景，總耗時加速 1.01×，最低場景比 0.99×。小圖使用相同原始繪製路徑，時間仍會有量測雜訊。 [資料](results/single_frame_small/metrics.json)

## 限制與研究判斷

主要限制仍是密集雲層的逐光線密度與陰影工作。當大部分區域必須細化，空間採樣省不了足夠工作；其餘已測試的捷徑不是變慢，就是消耗超過允許的品質誤差。這些實驗不能證明 10× 不可能，但目前不能宣稱已實現。

初始粗採樣看不到的細雲仍可能被漏掉；顏色梯度不是誤差上界。1440p 與 800×450 測試均加速，也不能推出所有視角、硬體、解析度都沒有降速。畫質參考是原 PSMS 近似 shader，不是收斂的物理體積散射；「肉眼不可分辨」尚無盲測證據。

GL_TIME_ELAPSED 涵蓋全部前置步驟、分類、重建、stencil 與 shade，但不含首次 shader 編譯、資源配置、讀回與 CPU 指標。Python／driver 提交空檔可能進入整段 GPU timer；逐 pass 計時條件不同，其加總不等於整段，不能用來替換主結果。沒有跨幀保留渲染結果。

## 重現

```bash
python 9_cascade_sky_rendering/program/benchmark.py --verify-selected --output verify_single_frame_final
python 9_cascade_sky_rendering/program/benchmark.py --research --experiment selected --variant-keys anchor,optimized --threshold .035 --width 2560 --height 1440 --frames 9 --output single_frame_final_1440 --profile --temporal
python 9_cascade_sky_rendering/program/benchmark.py --research --experiment selected --variant-keys anchor,optimized --threshold .035 --heldout --width 2560 --height 1440 --frames 7 --output single_frame_heldout_1440
MPLCONFIGDIR=/tmp/cascade_sky_mpl python 9_cascade_sky_rendering/program/analyze_single_frame.py
```

每個資料夾保留當時的 source snapshot、SHA-256、GPU、場景參數與每次計時，原研究資料未覆蓋。

## 相關方法

共用大氣查表的方向與 [Hillaire 2020](https://sebh.github.io/publications/egsr2020.pdf) 的方法相關；本實作保留 PSMS 的散射近似，沒有宣稱移植其完整模型。[Nubis, Evolved](https://www.guerrilla-games.com/read/nubis-evolved) 也說明單張高品質雲需要體積表示與採樣成本的整體設計。階層式體積採樣本身可追溯至 [Levoy 1990](https://graphics.stanford.edu/papers/Levoy-hpscans/adaptive-viscomp90/INDEX.HTM)，此處不主張基本思想的新穎性。
