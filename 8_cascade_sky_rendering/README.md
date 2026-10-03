# 9 — Cascade Sky Rendering：依影像梯度細化天空與體積雲

後續針對「單張、維持原品質、沒有場景降速、平均 10×」的研究，見
[V2 研究報告](V2_REPORT.md) 與 [V2 選定管線](program/single_frame_v2.py)。
V2 在原十四場景 1440p 實測 5.37×，同輪 V1 為 5.15×；採用先算雲、完全不透明時省略背景天空的改動。
[V1 研究報告](SINGLE_FRAME_REPORT.md) 與 [V1 管線](program/single_frame.py) 保留供比較。
**10× 目標尚未達成**；報告保留品質不合格的高速版本與其他失敗實驗，不將它們列為成功結果。
以下內容與 [REPORT.md](REPORT.md) 保留第一階段梯度階層實驗，原 `confirm_1440` 圖像不變。

本研究實作由低解析度到高解析度的多層渲染：只有上一層顏色梯度超過門檻的區域，才在下一層執行完整大氣與體積雲積分；其餘位置由上一層內插。這裡的「精度」指**空間解析度**，各層的 ray-march 品質參數保持相同。

- [實測結果與圖表](REPORT.md)
- [GPU 渲染器](program/cascade.py)、[梯度判斷與內插](program/reconstruct.frag)
- [可重現測試入口](program/benchmark.py)、[正確性驗證](program/verify.py)
- [800×450 掃描資料](results/sweep_800/metrics.json)
- [2560×1440 比較資料](results/confirm_1440/metrics.json)

## 與研究 0～3 的關係

基準為 `0_cloudworks/shader_HSE_PSMS_Next_2.glsl`，原始內容逐位元複製到 [baseline_psms.glsl](program/baseline_psms.glsl)，不修改研究 0～3。

它包含 CloudWorks/HSE 雲形、Next 自適應光線步進，以及源自 Papers 1～3 的 Chapman 光學厚度、大氣散射、雲照明和空氣透視。Paper 3 的 GLSL 多重散射本身是方向 gather 加高階增益的近似，並非論文 Python two-stream/SOS 求解器。本研究增加的是**螢幕空間採樣策略**，沒有把論文的物理誤差界當成 shader 或階層重建的誤差界。

為了使用 early stencil tests，wrapper 使用 OpenGL/GLSL 4.3，將原 shader 的保留字 `active` 重新命名，並將相機、太陽、視角偏移改成 uniform。原檔快照保持不變。另以原 GLSL 330 wrapper 驗證輸出差異。

## 演算法

令最終解析度為 \(W\times H\)，倍率為整數 \(N\ge2\)，層數為 \(L\)。層索引 \(\ell=0\ldots L-1\)，每層寬高為：

\[
(W_\ell,H_\ell)=\left(\left\lceil\frac{W}{N^{L-1-\ell}}\right\rceil,
\left\lceil\frac{H}{N^{L-1-\ell}}\right\rceil\right).
\]

預設 `N=2` 表示寬、高各增為兩倍，像素數約四倍。三層 2560×1440 為 640×360 → 1280×720 → 2560×1440。奇數尺寸也使用正確的最終畫面座標，不改變 FOV 或長寬比。

1. **初始層：** 每個像素完整執行原本的天空＋體積雲 shader，得到線性 HDR 色彩。
2. **每個細化層：** 找到上一層的雙線性內插 footprint。預設再向外擴張一個上一層像素，形成 4×4 採樣區；`radius=0` 則只檢查 2×2。
3. **梯度：** 對區域內水平、垂直相鄰樣本，計算顯示色彩 RGB 最大通道差的最大值：

   \[
   g_\ell=\max_{(a,b)\in E(\mathcal N)}\|D(C_{\ell-1}(a))-D(C_{\ell-1}(b))\|_\infty,
   \quad D(C)=[1-e^{-1.2\max(C,0)}]^{1/2.2}.
   \]

   這是上一層相鄰像素的有限差分，**沒有除以上一層像素在最終畫面的跨度**。各層使用同一門檻 \(T\)，所以會自然容許較平滑區域停留在較粗層；門檻數值與曝光／顯示轉換有關。
4. **重建：** 若 \(g_\ell>T\)，重新計算該細化像素的完整天空＋雲；否則使用上一層 HDR 雙線性內插。被細化的像素覆寫內插值，兩者不額外混合。

   \[
   C_\ell(x)=\begin{cases}
   R(x),&g_\ell(x)>T,\\
   \operatorname{bilinear}(C_{\ell-1},x),&\text{otherwise}.
   \end{cases}
   \]

這是每幀獨立的階層，沒有使用上一幀、temporal reprojection 或低品質 march。上一層可以包含更粗層內插而來的像素，這些結果同樣參與下一層梯度判斷。

## GPU 實作

每個細化層包含三個全螢幕 pass：

| Pass | 工作 |
|---|---|
| Reconstruct/classify | 寫入 HDR 內插色彩與 R8 細化遮罩 |
| Stencil | 依 R8 遮罩寫入 stencil，其他像素 discard |
| Shade | `layout(early_fragment_tests)` 讓 stencil 未通過的像素跳過完整天空／雲計算 |

遮罩建立和使用都留在 GPU，主流程不讀回遮罩、不由 CPU 選區。Stencil pass 使用只掛載 depth/stencil 的獨立 FBO，避免一邊讀遮罩、一邊把同一遮罩紋理掛在繪製 FBO 上。各層保留獨立 HDR 紋理，輸入與輸出不重疊。

內插用明確的浮點 `texelFetch`/`mix`，避免硬體紋理濾波的次像素權重量化。附件使用 RGBA32F + R8 + DEPTH24_STENCIL8；這是研究設定，尚未量測 RGBA16F 或遮罩壓縮。

## 效能假設與限制

若每次完整 shade 的成本相等，初始層必算、各細化層實際重算率為 \(p_\ell\)，則相對於全解析度的理想 shade 工作量為：

\[
Q=N^{-2(L-1)}+\sum_{\ell=1}^{L-1}p_\ell N^{-2(L-1-\ell)}.
\]

這是工作量指標，**不是預測 GPU 加速比的公式**。雲邊緣的光線本身可能比較貴，零碎遮罩影響 SIMD 利用率，各層也會重複 shade。當所有像素都細化，`N=2,L=3` 已是 1.3125 倍的像素積分量，另有分類、內插、stencil、頻寬和命令提交成本。

單靠已採樣色彩梯度不能保證抓到尚未採到的細節：如果細小雲絲、亮點或太陽完全落在初始採樣點之間，上一層可能是平的，後續永遠不會要求細化。驗證程式保留一個能重現此情況的單像素亮點案例。4×4 halo 只能擴張**已看見**的邊緣，無法修復這個資訊缺失；後續可研究強制探索採樣、深度／光學厚度輔助判斷或保守幾何遮罩。

## 比較方法

基準與階層使用相同 shader、最終解析度、相機、時間、太陽、march 參數及顯示轉換。每個場景先預熱，每次重複輪替／反轉 variant 順序，記錄 `GL_TIME_ELAPSED` 原始樣本、median 和 p95。主要計時涵蓋初始層及所有細化 pass，排除編譯、配置、圖像讀回、CPU 指標計算和 PNG 輸出。

完整多 pass timer 也可能包含 GPU 等候 Python/driver 提交命令的空檔。因此另記錄各 draw 的 GPU query 耗時，供辨認成本來源；其加總不是引擎整合後的實測幀時間，不能拿來取代主結果。

使用研究 0 的十二個固定場景（雲海／積雨雲，各三視角，加六種日落前後光照），另加兩個純大氣場景。純大氣場景停用雲的合成，用來檢查低變化天空的收益。畫面地平線下為原 shader 的黑色背景，並非真實引擎地形；它也會影響邊緣與內插誤差。

品質參考是**同一 shader 的逐像素全解析度畫面**，沒有重跑高精度物理積分。指標量測階層額外引入的重建誤差：顯示 RGB RMSE、PSNR、luminance SSIM、邊緣梯度與邊緣區 RMSE、p99／最大通道誤差、誤差 >0.03 的像素比例，以及 HDR 相對 RMSE。

本次操作性品質門檻為所有場景同時滿足：RMSE ≤0.005、SSIM ≥0.99、最大通道誤差 >0.03 的像素 ≤1%。這不是視覺無損或普遍誤差保證。參數是在這批場景上探索，1440p 是解析度複驗，並非獨立場景驗證集。

另比較一幀 1/60 秒的雲時間變化及相機平移 10 公尺前後的**重建誤差變化**：

\[
\operatorname{RMSE}\big[(C_{t+1}-R_{t+1})-(C_t-R_t)\big].
\]

其中色彩先經顯示轉換。此診斷把原場景本身的變化扣除，仍不是完整動畫的時域穩定性評估。

## 重現

需求：可存取 NVIDIA/其他硬體 OpenGL 4.3 的桌面或離屏顯示環境、Python 3.9+、NumPy、SciPy、Pillow、GLFW、PyOpenGL。程式拒絕把 llvmpipe/softpipe 的時間當成硬體 GPU 結果。

以下命令從 `articles/` 執行：

```bash
python 9_cascade_sky_rendering/program/benchmark.py --verify --output verification
python 9_cascade_sky_rendering/program/benchmark.py --width 800 --height 450 --frames 7 --preset sweep --output sweep_800 --temporal
python 9_cascade_sky_rendering/program/benchmark.py --width 2560 --height 1440 --frames 9 --preset confirm --output confirm_1440 --temporal --profile
python 9_cascade_sky_rendering/program/benchmark.py --width 810 --height 459 --frames 7 --factor 3 --levels 3 --threshold 0.04 --scenes A-g,B-t,sky-day --output factor3_check
python 9_cascade_sky_rendering/program/analyze.py
```

`--factor 3` 可改倍率；`--scenes A-g,B-t,sky-day` 可縮小場景集合；`--levels 3 --threshold 0.04 --radius 1` 可指定一組自訂階層，與原版比較。`--preset smoke` 供快速檢查，低解析度冒煙測試不當成正式效能結論。每次結果保留解析度、GPU、參數、原始計時樣本及源碼 SHA-256。正式 1440p 測量版本的原始碼另保留在 `results/confirm_1440/source/`。

輸出中 `uniform-half`／`uniform-quarter` 是同一階層管線強制只內插的消融對照，分別從 final/N、final/N² 起算；N≠2 時名字僅是歷史標籤。它們仍包含遮罩與空的細化 draw 成本，並非最精簡的傳統 upscale 實作。

目前是獨立 GPU 研究 harness，尚未接入飛行模擬器的實際 frame graph。專案上層既有 `.gitignore` 會忽略 `articles/*`；本研究檔案保存在此目錄，沒有修改該規則或自動提交 Git。
