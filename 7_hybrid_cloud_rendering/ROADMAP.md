可以，而且我建議正式把研究主線從：

> **A3 作為完整精確雲 renderer**

改成：

> **A3/B 作為公里尺度的 transport-aware interval accelerator；傳統短程 raymarch 負責邊界品質；tiny MLP 負責低維 transport closure。**

這不是退讓，而是重新把 A3 放到它最有優勢的位置：**積雨雲、超大垂直體積、長空路徑、稀疏 active regions**。

以下是一條可以直接交給 local agent 分工實作的完整路徑。

---

# 一、研究目標與成功條件

## 核心研究問題

在公里尺度、垂直發展的 procedural cumulonimbus 場景中，能否利用：

1. A3/B0/B1/B2 找出有效體積區段；
2. 短程 raymarch 重建輪廓與高頻密度；
3. tiny MLP 預測區段級散射 closure；
4. 時域與空間重建輸出 4K；

使 RTX 4070 達到：

[
3840\times2160,\qquad \geq 30\ \mathrm{FPS}
]

即總 frame budget：

[
33.33\ \mathrm{ms}
]

其中建議雲系統預算：

[
18\text{–}23\ \mathrm{ms}
]

保留約 10–15 ms 給場景、後處理和 UI。

## 最低論文成功標準

不要求 hybrid 在所有場景擊敗傳統 raymarcher，而是要求：

* 在大尺度積雨雲場景顯著降低 density evaluation；
* final 4K output 達到 30 FPS；
* 在相同時間下，品質高於傳統 raymarch；
* 或在相同品質下，速度至少快 1.5–2 倍；
* 對平層雲不應比傳統方法慢超過 15%；
* MLP 不應破壞 temporal stability；
* A3/B 必須能在不需要神經網路時單獨工作。

---

# 二、整體系統設計

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Procedural Cloud Definition                       │
│ weather field / macro SDF / FBM / erosion / height / animation      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                 ┌─────────────▼─────────────┐
                 │ Macro Support Hierarchy   │
                 │ A3 bounds + B statistics  │
                 │ μ, σ², cumulants, support │
                 └─────────────┬─────────────┘
                               │
Camera ray                     ▼
────────────────────► Macro Interval Traversal
                               │
                  ┌────────────┴────────────┐
                  │ Active interval stream  │
                  │ [t0,t1,type,features]   │
                  └────────────┬────────────┘
                               │
           ┌───────────────────┼─────────────────────┐
           │                   │                     │
           ▼                   ▼                     ▼
    Empty / trivial      Smooth interior       Complex boundary
       skip              B1/B2 closure         short raymarch
                                                   │
                                                   ▼
                                          density + single scatter
           └───────────────────┬─────────────────────┘
                               ▼
                    Tiny Neural Closure
                    MS residual / correction
                               │
                               ▼
                    Interval compositing
                               │
                               ▼
                    Low-resolution cloud buffer
                               │
                               ▼
          Temporal accumulation + edge-aware reconstruction
                               │
                               ▼
                         4K final image
```

---

# 三、子系統總覽

系統分成六個子系統。

| 子系統                    | 職責                               |              執行頻率 |
| ---------------------- | -------------------------------- | ----------------: |
| S0 Procedural Field    | 定義巨觀與高頻雲密度                       |  sample-dependent |
| S1 A3/B Hierarchy      | 空域排除、interval 建構、統計              |             每 ray |
| S2 Interval Classifier | 決定 skip／analytic／march／MLP       |        每 interval |
| S3 Short Raymarch      | 邊界、高頻、silver lining              |       少數 interval |
| S4 Tiny MLP Closure    | 多重散射與解析誤差修正                      | 每 active interval |
| S5 Reconstruction      | temporal accumulation、4K upscale |           每 pixel |

---

# 四、S0：程序雲場設計

## 4.1 密度結構

建議初期固定使用一個足夠傳統、但可分析的 procedural field：

[
\rho(\mathbf{x},t)
==================

\rho_{\max}
\operatorname{smoothReLU}
\left[
M(\mathbf{x},t)
+
D(\mathbf{x},t)
---------------

\theta
\right]
]

其中：

[
M =
W(\mathbf{x}*{xz},t)
H(y)
+
A*{\mathrm{tower}}(\mathbf{x},t)
]

[
D =
\sum_{k=0}^{K-1} a_k
N_k(2^k\mathbf{x}+\mathbf{v}_k t)
]

組件：

* `W`：低頻 weather field；
* `H`：高度 profile；
* `A_tower`：積雨雲 tower macro field；
* `N_k`：Perlin/Worley/wave noise octave；
* smoothReLU：避免完全不可導的 hard threshold；
* 後續再加入 erosion、curl warp、min/max。

## 4.2 第一版限制

第一版不要一開始就支援所有 operator。

先支援：

```text
add
weighted_sum
multiply_by_smooth_mask
smoothstep
smooth_ReLU
affine_coordinate_transform
FBM octave sum
```

第二階段才加入：

```text
hard threshold
max/min
erosion subtraction
domain warp
curl warp
nested cellular noise
```

原因是 B1/B2 統計 closure 對非線性 operator 非常敏感。

## 4.3 Procedural noise 論文參考

Local agent 應先讀：

### P1 — Fast Sphere Tracing of Procedural Volumetric Noise for Very Large and Detailed Scenes

重點讀取：

* FBM 如何表示成 nested bounding volumes；
* derivative bounds；
* octave truncation；
* large-scene traversal；
* 哪些 bound 可安全跳步；
* 是否只解 surface intersection，還是適用 density support。

這篇是 A3 的直接相鄰工作，但其主要目標是程序噪聲與 implicit geometry 的 sphere tracing，不是保存 optical-depth/free-path statistics。([Inria HAL][1])

### P2 — Multi-Dimensional Procedural Wave Noise

評估 wave basis 是否比 Perlin/Worley 更容易：

* 求局部 amplitude bound；
* 求方向頻譜；
* 截斷高頻；
* 保存 variance；
* shader 計算。

該工作提出可在多維度使用且具有頻譜控制的程序 wave noise。([Pascal Guehl][2])

---

# 五、S1：A3/B Macro Interval Traversal

## 5.1 A3 在新系統中的角色

A3 不再求最終 collision event，也不必對每個 cell 執行完整 degree-10 Newton。

它只需要產生：

```cpp
struct CloudInterval
{
    float t0;
    float t1;

    float tauMean;
    float tauVariance;

    float densityMin;
    float densityMax;
    float densityMean;
    float densityVariance;

    float thresholdMargin;
    float unresolvedEnergy;

    uint activeOctaveMask;
    uint intervalClass;
};
```

## 5.2 A3-Lite

從目前 A3-GBox64/F32 分支出一個 `A3-Lite`：

### 保留

* FP64 camera ray normalization；
* FP64 slab/volume entry；
* transactional DDA；
* multiscale hierarchy；
* conservative interval bounds；
* deterministic traversal；
* stable grazing handling。

### 移除或降級

* 不對所有 interval 求精確 collision；
* 不逐 cell 完整求 degree-10 root；
* 不在宏觀 traversal 做完整 radiance integration；
* 只在需要的 interval 計算高階係數；
* 統計不足時標記 `uncertain`，交給 raymarch。

## 5.3 Interval termination 條件

每一 hierarchy node 建立：

[
\rho_{\min},\rho_{\max},\mu_\rho,\sigma_\rho^2
]

如果：

[
\rho_{\max}<\epsilon_{\rm empty}
]

則跳過。

如果：

[
(\rho_{\max}-\rho_{\min})L < \epsilon_\tau
]

則可視為 smooth interval。

如果：

[
\sigma_\rho^2 L^2 < \epsilon_{\rm variance}
]

可使用 B1。

如果 variance 不可忽略，但高階 cumulant 足夠穩定，使用 B2。

否則：

```text
complex interval → short raymarch
```

---

# 六、B0、B1、B2 的具體定義

## 6.1 B0：Raw Cutoff

截斷高頻 octave：

[
\rho_{B0}
=========

f\left(
\sum_{k=0}^{K_c}a_kN_k
\right)
]

未解析高頻直接忽略。

用途：

* baseline；
* 空域排除；
* 遠距離或極低 opacity interval；
* 評估「只做 coarse octave」會犯多少 transport error。

輸出：

* coarse density；
* conservative unresolved amplitude；
* unresolved octave energy。

## 6.2 B1：Mean Compensation

對未解析高頻加入期望修正：

[
\rho_{B1}
=========

E\left[
f(X+Y)
\mid X
\right]
]

其中：

* (X)：已解析低頻；
* (Y)：未解析高頻。

線性情況可以簡化，但 threshold/ReLU 後不能只使用 (f(X+E[Y]))。

對 Gaussian-like residual，可推導：

[
E[\max(0,X+Y-\theta)]
]

的解析式或 LUT。

B1 應儲存：

* residual mean；
* residual variance；
* threshold margin；
* operator type。

## 6.3 B2：Variance/Cumulant Compensation

目標不是只保存平均 density，而是近似保存：

[
E[e^{-\tau}]
]

因為一般而言：

[
E[e^{-\tau}]
\neq
e^{-E[\tau]}
]

利用 cumulant expansion：

[
\log E[e^{-\tau}]
=================

-\kappa_1
+\frac{\kappa_2}{2}
-\frac{\kappa_3}{6}
+\frac{\kappa_4}{24}
-\cdots
]

第一版保留：

[
\kappa_1,\kappa_2
]

第二版測試：

[
\kappa_3,\kappa_4
]

定義 effective optical depth：

[
\tau_{\mathrm{eff}}
===================

\kappa_1-\frac{\kappa_2}{2}
+\frac{\kappa_3}{6}
-\frac{\kappa_4}{24}
]

但必須注意：

* 此展開只在相應分布與尺度範圍內可靠；
* 不保證對所有高 variance interval 收斂；
* 因此必須存在 fallback gate。

## 6.4 B2 不應被強迫處理所有 interval

B2 適合：

* interior；
* unresolved fluctuations 相對均勻；
* 累積路徑較長；
* threshold 不落在極敏感區。

不適合：

* silhouette；
* 雲洞；
* threshold crossing；
* density discontinuity；
* domain warp fold；
* 強銀邊。

這些直接 dispatch 到 raymarch。

---

# 七、S2：Interval Classifier

最初使用 deterministic classifier，不要先用神經網路。

```cpp
if (densityMax < emptyThreshold)
    return EMPTY;

if (projectedLength < minIntervalLength)
    return SHORT_MARCH;

if (thresholdMargin < boundaryMargin)
    return SHORT_MARCH;

if (unresolvedEnergy < b1Limit)
    return B1_ANALYTIC;

if (varianceNormalized < b2Limit &&
    cumulantRisk < cumulantLimit)
    return B2_ANALYTIC;

return SHORT_MARCH;
```

## 7.1 分類器輸入

* interval world length；
* projected pixel length；
* density min/max；
* mean/variance；
* unresolved octave energy；
* threshold margin；
* distance to camera；
* view–sun cosine；
* estimated optical depth；
* temporal history confidence；
* current GPU quality tier。

## 7.2 後續可選 learned classifier

只有當 deterministic classifier 過於保守時，才訓練一個 classifier 預測：

[
P(\text{analytic error}<\epsilon)
]

輸出不是 radiance，而是：

```text
safe probability
recommended mode
sample budget
```

這可以很小，例如：

[
16\rightarrow16\rightarrow4
]

---

# 八、S3：Short-Interval Raymarch

## 8.1 Raymarch 只處理 active interval

不是沿完整公里級 ray march，而是在 A3 找出的區段中執行。

例如完整 ray：

```text
0 km ───────────────────────────── 16 km
       empty      cloud      empty
                   │
                   ▼
                 250 m active interval
```

raymarch 只處理 250 m。

## 8.2 Sample budget

每 interval：

| 分類              | Camera samples | Sun samples |
| --------------- | -------------: | ----------: |
| Boundary low    |              4 |           2 |
| Boundary medium |              8 |         3–4 |
| Boundary high   |          12–16 |         4–6 |
| Dense interior  |            0–4 |         0–2 |
| Analytic B1/B2  |              0 |           0 |

每 ray 建議上限：

* camera samples：24；
* sun samples：12；
* active intervals：4；
* density evaluations：48–64。

若超出則：

* 降低 octave；
* merge interval；
* 使用 analytic closure；
* 或降低 quality tier。

## 8.3 Step placement

使用：

* blue-noise initial jitter；
* stratified sampling；
* density-gradient biased sampling；
* boundary 前後加密；
* interior 疏化；
* temporal sample rotation。

## 8.4 Lighting baseline

先使用：

[
L_{\mathrm{base}}
=================

L_{\mathrm{single}}
+
L_{\mathrm{cheapMS}}
]

其中 cheap multiple scattering 可以是：

* octave approximation；
* two-lobe phase；
* two-stream；
* Beer-powder；
* energy-conserving empirical term。

論文中建議以 pseudo-spherical two-stream 作為主要物理 baseline，而 empirical method 作遊戲 baseline。

---

# 九、S4：Tiny MLP Closure

## 9.1 第一個 MLP 只做 multiple-scattering residual

不要讓第一版網路預測完整 radiance。

定義：

[
\Delta\mathbf L
===============

## \mathbf L_{\mathrm{reference}}

\mathbf L_{\mathrm{base}}
]

最終：

[
\mathbf L_{\mathrm{hybrid}}
===========================

\mathbf L_{\mathrm{base}}
+
w_{\mathrm{confidence}}\Delta\mathbf L_{\mathrm{MLP}}
]

這使網路只學習：

* high-order scattering；
* angular residual；
* B2 closure error；
* local shadow leakage；
* silver-lining correction。

相較於直接學完整光照，residual 學習通常更容易保留基本物理關係。

## 9.2 網路結構

第一版：

[
24\rightarrow32\rightarrow32\rightarrow4
]

輸出：

```text
RGB residual
confidence / residual scale
```

MAC 約：

[
24(32)+32(32)+32(4)=1920
]

權重數約 2000 個，FP16 約 4 KB。

第二版候選：

[
32\rightarrow32\rightarrow32\rightarrow8
]

仍控制在約 2,500 MAC。

NVIDIA 已經提供將小型網路放入 shader 並使用 cooperative vectors 執行矩陣向量運算的工具鏈；其公開資料說明這類功能已用於 RTX Neural Shaders，並能從 shader 使用 Tensor Core。([NVIDIA Developer][3])

## 9.3 MLP 輸入特徵

### 幾何特徵

1. interval length；
2. camera distance；
3. normalized cloud height；
4. view–sun cosine；
5. view–vertical cosine；
6. sun–vertical cosine。

### 光學特徵

7. mean optical depth；
8. effective B2 optical depth；
9. density mean；
10. density variance；
11. third cumulant；
12. fourth cumulant；
13. single-scattering albedo；
14. phase asymmetry (g)；
15. direct sun transmittance；
16. coarse ambient irradiance。

### 結構特徵

17. threshold margin；
18. unresolved octave energy；
19. boundary probability；
20. normalized density gradient；
21. entry density；
22. exit density；
23. active octave count；
24. interval class。

不要輸入絕對世界座標，避免網路記住特定雲。

## 9.4 特徵正規化

所有輸入離線計算 normalization：

```python
x_norm = clamp((x - mean) / std, -4, 4)
```

但 optical depth 建議：

[
x_\tau=\log(1+\tau)
]

variance 建議：

[
x_{\sigma^2}=\log(\epsilon+\sigma^2)
]

角度直接使用 cosine。

## 9.5 Output parameterization

不要直接預測任意 signed RGB。

建議：

[
\Delta L_c
==========

s_c
\tanh(y_c)
]

其中 (s_c) 是依 baseline radiance 或 sun radiance 決定的安全 scale。

或預測 multiplicative correction：

[
L_c
===

L_{\mathrm{base},c}
\exp(\delta_c)
]

但銀邊／暗部可能同時需要 additive residual，因此第一版建議：

[
L=L_{\mathrm{base}}+
s_{\mathrm{add}}\tanh(\Delta)
]

## 9.6 Shader 路徑

### Portable path

* HLSL/GLSL；
* FP16 weight；
* unrolled `dot`；
* ReLU 或 leaky ReLU；
* constant buffer weights；
* 每 interval 一次。

### RTX path

* Shader Model 6.9 cooperative vectors；
* FP16；
* Tensor Core；
* Slang/HLSL；
* fallback 到普通 shader ALU。

Cooperative-vector API 的設計目的正是讓 shader 中的小型 MLP layer 以矩陣向量乘法形式映射至 GPU 矩陣硬體。([NVIDIA Developer][3])

---

# 十、Ground-truth 資料集建構

## 10.1 Reference solver

建立三個品質層：

### GT-A：Path tracing

* volumetric path tracing；
* next-event estimation；
* MIS；
* Russian roulette；
* 高 sample count；
* 真正多重散射。

### GT-B：PSRT/SOS

* spherical shell geometry；
* SOS multiple scattering；
* 作為可驗證 reference；
* 適合大氣與 twilight case。

### GT-C：High-quality deterministic raymarch

* 512–2048 camera samples；
* 128–512 sun samples；
* high-quality scattering approximation；
* 用於快速產生 preliminary dataset。

訓練第一版可使用 GT-C 預訓練，最後用 GT-A/GT-B fine-tune。

近期神經體積散射工作也採用 density、transmittance 與 phase-related features，透過網路近似昂貴的 RTE 多重散射過程；但其 3D CNN/attention 架構比本方案預定的 interval-level tiny MLP 更重。([arXiv][4])

## 10.2 Dataset unit

不要以完整 4K image 作為主要訓練樣本。

基本樣本單位是一個 `interval record`：

```python
IntervalSample:
    procedural_config
    t0, t1
    ray_origin
    view_direction
    sun_direction

    B0_features
    B1_features
    B2_features

    baseline_single_scattering
    baseline_two_stream
    reference_radiance

    reference_transmittance
    reference_multiple_scattering
    error_class
```

## 10.3 場景參數採樣

### 巨觀雲型

至少建立：

* isolated cumulonimbus；
* multi-cell cumulonimbus；
* squall line；
* towering cumulus；
* anvil cloud；
* broken cumulus；
* stratocumulus control group；
* thin cirrus negative/control group。

### 尺度

* 寬度：1–20 km；
* 高度：0.5–15 km；
* camera distance：0.1–100 km；
* volume world scale：至少三個 orders。

### Optical parameters

* extinction scale；
* albedo；
* HG (g)；
* density scale；
* sun elevation；
* sky brightness；
* ground albedo。

### 程序參數

* octave count；
* lacunarity；
* persistence；
* threshold；
* erosion；
* warp magnitude；
* advection velocity；
* tower aspect ratio；
* anvil strength。

## 10.4 Sample distribution

第一階段：

* 500 procedural cloud configurations；
* 每個配置 64 camera poses；
* 每 pose 128–512 interval samples；
* 約 5–15 million interval records。

不要儲存所有 raw density samples。儲存：

* compact features；
* target residual；
* scene/config ID；
* difficulty metadata。

估計每 record 128–256 bytes，10 million records 約 1.3–2.6 GB。

## 10.5 分割方式

不能隨機把同一朵雲的 rays 分散到 train/test。

應以 procedural configuration 分割：

* train：70% 雲配置；
* validation：15%；
* test：15%。

額外建立 OOD：

* 未見 octave count；
* 未見 scale；
* 未見 threshold；
* 未見 sun elevation；
* 未見 cloud morphology；
* 未見 procedural operator。

這才能測試網路是否學會 transport closure，而不是記住場景。

## 10.6 Hard-example mining

訓練後重新跑 renderer，收集：

* 最大 HDR relative error；
* silhouette error；
* silver-lining error；
* dark-core error；
* twilight error；
* temporal flicker；
* classifier false-safe。

將這些 interval 加入下一輪訓練。

---

# 十一、MLP 訓練方法

## 11.1 Loss function

主 loss：

[
\mathcal L_{\mathrm{HDR}}
=========================

\left|
\log(1+\alpha L_{\mathrm{pred}})
--------------------------------

\log(1+\alpha L_{\mathrm{GT}})
\right|_1
]

附加 relative loss：

[
\mathcal L_{\mathrm{rel}}
=========================

\frac{
|L_{\mathrm{pred}}-L_{\mathrm{GT}}|*1
}{
\epsilon+|L*{\mathrm{GT}}|_1
}
]

總 loss：

[
\mathcal L
==========

\lambda_1\mathcal L_{\mathrm{HDR}}
+
\lambda_2\mathcal L_{\mathrm{rel}}
+
\lambda_3\mathcal L_{\mathrm{energy}}
+
\lambda_4\mathcal L_{\mathrm{confidence}}
]

其中 energy loss 限制：

* radiance 非負；
* 不無限制超過 incident energy；
* 高 optical depth 下輸出平滑；
* 無散射時 residual 趨近零。

## 11.2 Confidence head

輸出第四維：

[
c\in[0,1]
]

訓練 target：

[
c^*=
\exp(-\beta e)
]

其中 (e) 是預測相對誤差。

runtime：

```cpp
if (confidence > threshold)
    use neural residual;
else
    use conservative fallback;
```

但 confidence 容易過度自信，因此需做：

* temperature scaling；
* validation calibration；
* reliability diagram；
* OOD threshold。

## 11.3 Optimizer

第一版：

* AdamW；
* batch size 8192–65536 interval records；
* learning rate (10^{-3})；
* cosine decay；
* weight decay (10^{-6}) 至 (10^{-5})；
* 50–200 epochs；
* early stopping。

## 11.4 Quantization

訓練順序：

1. FP32；
2. FP16 inference validation；
3. FP16 export；
4. optional INT8 quantization-aware training。

第一篇論文建議以 FP16 為主，不要讓 INT8 deployment 成為主要風險。

---

# 十二、S5：4K 重建設計

真正 4K 每 pixel 做完整 volume rendering，RTX 4070 很難留下足夠場景預算。

應定義為：

> **4K final output quality，而非所有 volume samples 原生 4K。**

## 12.1 Internal resolution

提供三種模式：

| 模式          | Cloud internal resolution | 適用          |
| ----------- | ------------------------: | ----------- |
| Quality     |                 1920×1080 | 靜態、高品質      |
| Balanced    |                  1280×720 | 主要 4K/30 目標 |
| Performance |                   960×540 | 高雲量或快速移動    |

推薦主實驗：

[
1280\times720
\rightarrow
3840\times2160
]

配合：

* temporal accumulation；
* motion vectors；
* cloud depth；
* transmittance；
* normal/gradient proxy；
* interval ID；
* history rejection。

## 12.2 G-buffer

Cloud pass 輸出：

```text
RGB radiance
transmittance
representative depth
depth variance
density gradient
motion vector
history confidence
interval complexity
```

## 12.3 Temporal reconstruction

history weight 應受：

* camera motion；
* cloud velocity；
* depth disagreement；
* transmittance disagreement；
* density-gradient disagreement；
* lighting change；
* procedural phase change；

控制。

避免使用只依靠 screen-space depth 的普通 TAA，因為體積沒有單一表面深度。

## 12.4 Quality validation

同時報告：

* raw internal-resolution output；
* spatial-only upscale；
* temporal upscale；
* native 4K reference。

否則無法判斷性能來自算法還是單純降解析度。

---

# 十三、GPU 實作分工

## Pass 0：Cloud hierarchy update

頻率：

* 每幀只更新移動區域；
* 或每 2–8 幀更新 coarse hierarchy；
* noise 本身仍 procedural，不建立完整 voxel grid。

輸出：

* macro support cells；
* bounds；
* moments；
* active octave metadata。

## Pass 1：Macro traversal

每 internal-resolution pixel 一個 thread：

* volume entry；
* A3-Lite traversal；
* 產生最多 4 個 active intervals；
* interval 資料保存在 register/local structure；
* 避免全畫面 global interval list。

## Pass 2：Interval integration

在同一 shader 或第二 compute pass：

* analytic B1/B2；
* short raymarch；
* sun query；
* MLP inference；
* front-to-back compositing。

第一版應避免 persistent global queue，除非 divergence 太嚴重。

## Pass 3：Temporal reconstruction

* reprojection；
* history rejection；
* accumulation；
* spatial upscale；
* final compositing。

---

# 十四、RTX 4070 4K/30 FPS 預算

以下是**設計目標，不是已測得結果**。

## Balanced 模式預算

| Pass                         |             目標時間 |
| ---------------------------- | ---------------: |
| Macro hierarchy/update       |       0.5–1.5 ms |
| A3-Lite traversal            |       2.0–4.0 ms |
| Active interval raymarch     |       5.0–8.0 ms |
| Sun/lighting queries         |       3.0–5.0 ms |
| Tiny MLP                     |       0.5–1.5 ms |
| Temporal + 4K reconstruction |       2.0–4.0 ms |
| Composite                    |       0.3–0.8 ms |
| **Cloud total**              | **13.3–24.8 ms** |

主要目標：

[
T_{\mathrm{cloud,p95}} < 22\ \mathrm{ms}
]

而不是只報平均值。

## 關鍵 workload gate

在 1280×720 internal resolution 下：

* 平均 active intervals/ray：≤ 2；
* p95 active intervals/ray：≤ 4；
* 平均 full density samples/ray：≤ 16；
* p95：≤ 32；
* 平均 sun density samples/ray：≤ 8；
* MLP evaluations/ray：≤ 2；
* MLP evaluation：≤ 2,500 MAC；
* empty rays：A3-Lite 成本應低於 16–24 cheap procedural evaluations。

---

# 十五、實驗設計

## E0：傳統 renderer baseline

建立至少三個 baseline：

### RM-Fixed

* fixed step；
* standard short sun march；
* temporal upscale。

### RM-Adaptive

* empty-space coarse stepping；
* adaptive step；
* density mip；
* temporal jitter；
* early termination；
* heuristic multiple scattering。

### RM-Production

把你認為「效果和品質上勝出」的所有合理 heuristic 都加進來。

這是最重要的 baseline。不能只和弱版 raymarch 比。

---

## E1：A3-Lite traversal 能否處理公里級空域

場景：

* 2 km、5 km、10 km、20 km 積雨雲；
* camera ray path 5–100 km；
* 雲占 path 比例 1%–50%。

測量：

* traversal time；
* density evaluation count；
* active interval count；
* false-empty rate；
* missed optical depth；
* grazing stability。

預期：

* 稀疏度越高，A3-Lite 優勢越大；
* 平層雲優勢小；
* 密集全域霧可能完全沒有優勢。

---

## E2：B0/B1/B2 transport accuracy

比較：

```text
Full octave reference
B0
B1
B2-variance
B2-cumulant4
```

指標：

* (|\Delta\tau|)；
* transmittance relative error；
* free-path CDF Wasserstein distance；
* opacity error；
* interval radiance error；
* silhouette shift。

場景分組：

* interior；
* boundary；
* threshold-sensitive；
* low density；
* high optical depth；
* long interval。

預期：

* B0 在 threshold 邊界明顯失敗；
* B1 改善 mean opacity；
* B2 對長路徑 transmittance 改善；
* boundary 仍需 raymarch。

---

## E3：Hybrid dispatch ablation

比較：

1. 全 raymarch；
2. A3 + 全 active interval raymarch；
3. A3 + B1 + raymarch；
4. A3 + B2 + raymarch；
5. A3 + B2 + raymarch + MLP；
6. A3 + B2 + MLP without raymarch；
7. oracle classifier。

目的：

* 確認效能來自哪一層；
* 檢查 MLP 是否真的必要；
* 量化 classifier 保守程度；
* 找到 quality/performance Pareto front。

---

## E4：MLP architecture ablation

比較：

| 網路       |  參數級別 |
| -------- | ----: |
| Linear   | 約 100 |
| 16×16    | 約 700 |
| 32×32    |  約 2k |
| 32×32×32 |  約 3k |
| 64×64×64 | 約 11k |

測量：

* HDR PSNR；
* FLIP；
* temporal error；
* MLP runtime；
* shader register count；
* occupancy；
* FP16 vs FP32；
* cooperative-vector vs normal ALU。

---

## E5：MLP target ablation

比較預測：

1. full radiance；
2. multiple scattering only；
3. residual against single scattering；
4. residual against two-stream；
5. multiplicative correction；
6. additive correction。

預期最好的通常會是：

> two-stream／cheap-MS baseline + additive residual。

這是研究假設，需要實驗驗證。

---

## E6：泛化實驗

訓練未見：

* world scale；
* procedural seed；
* octave count；
* noise family；
* sun angle；
* extinction；
* (g)；
* cloud morphology。

特別重要的是：

### Scale generalization

同一網路測：

* 500 m；
* 2 km；
* 10 km；
* 20 km。

所有物理輸入必須使用 dimensionless 或 optical quantities，例如：

* optical depth；
* normalized height；
* normalized variance；
* angular cosine；

不能只輸入公尺數。

---

## E7：Temporal stability

建立 camera sequences：

* slow pan；
* high-speed fly-through；
* cloud edge crossing；
* sunrise/sunset；
* moving procedural noise；
* sudden lighting change。

指標：

[
E_t=
|
I_t-\mathcal W(I_{t-1})
|
]

同時測：

* flicker；
* ghosting；
* interval mode switching；
* MLP confidence switching；
* B2/raymarch transition seam。

必須加入 hysteresis：

```text
analytic → raymarch threshold A
raymarch → analytic threshold B
A != B
```

避免 mode 每幀震盪。

---

## E8：4K 性能壓力測試

至少測：

### Scene A：遠距 isolated cumulonimbus

* 大量空域；
* 預期 hybrid 最強。

### Scene B：camera inside cumulonimbus

* active volume 高；
* 預期 hybrid 優勢下降。

### Scene C：full-screen overcast

* 幾乎無空域；
* 作為負例。

### Scene D：multi-cell storm

* 多 interval；
* traversal divergence 壓力。

### Scene E：sunset silhouette

* 強相位、高 dynamic range；
* MLP 與 B2 壓力。

### Scene F：高速飛行

* temporal reconstruction 壓力。

報告：

* average；
* median；
* p95；
* p99；
* worst frame；
* VRAM；
* occupancy；
* wave divergence；
* density evaluations；
* MLP calls。

---

# 十六、品質指標

不能只報 PSNR。

## Image metrics

* HDR-VDP 或 HDR image metric；
* FLIP；
* log-PSNR；
* SSIM；
* LPIPS，僅作輔助；
* edge-weighted error；
* temporal warping error。

## Transport metrics

* optical-depth error；
* transmittance error；
* free-path CDF；
* interval integrated radiance；
* energy conservation；
* angular scattering error。

## Perceptual regions

分開報：

* silhouette；
* silver lining；
* dark core；
* sun-facing edge；
* shadowed interior；
* thin wisps；
* sky compositing boundary。

---

# 十七、預期結果

以下是合理目標，不是保證值。

## 17.1 遠距大型積雨雲

預期：

* 70–95% ray path 可被宏觀跳過；
* density evaluations 比 production raymarch 降低 2–6 倍；
* cloud pass 約 12–20 ms；
* 4K final output 30–50 FPS；
* hybrid 比純 B2 顯著改善輪廓；
* MLP 改善內部 whiteness 與散射層次。

## 17.2 多 cell storm

預期：

* density evaluations 降低 1.5–3 倍；
* 20–28 ms cloud pass；
* 可維持約 30 FPS；
* interval count 與 divergence 成為瓶頸。

## 17.3 Camera inside cloud

預期：

* A3 空域跳躍優勢較小；
* B1/B2 interior closure 仍可能降低 samples；
* 性能接近或略優於 production raymarch；
* 可能需要動態切換成 conventional mode。

## 17.4 Overcast/slab cloud

預期：

* A3 幾乎沒有 traversal 優勢；
* 系統應自動選擇純 raymarch path；
* overhead 目標低於 10–15%。

因此 renderer 應先做 scene-level dispatch：

```cpp
if (estimatedOccupancy > denseModeThreshold)
    useProductionRaymarch();
else
    useA3Hybrid();
```

---

# 十八、論文主張形式

最強的主張不應是：

> A3 比 raymarching 更快。

而應是：

> We introduce a scale-separated hybrid renderer for kilometer-scale procedural clouds. A transport-aware macro traversal extracts active optical intervals, analytic statistical closures handle homogeneous interiors, short-range ray marching preserves high-frequency boundaries, and a shader-resident neural residual estimates unresolved multiple scattering.

可以拆成四個貢獻：

1. **Macro optical interval traversal**
   將公里級程序介質轉成少量 active transport intervals。

2. **Transport-aware procedural reduction**
   B1/B2 保存 mean transmittance 與 cumulant effects，而不是只保存 density。

3. **Adaptive hybrid integration**
   analytic interior + short boundary raymarch。

4. **Shader-resident neural scattering closure**
   小型 MLP 每 interval 執行，不在每 density sample 執行。

---

# 十九、Local agent 閱讀清單

## A3／程序 traversal

### 必讀 A

**Fast Sphere Tracing of Procedural Volumetric Noise for Very Large and Detailed Scenes**

要求 agent 回答：

* nested FBM bounds 如何建構；
* 一階／二階 derivative bound；
* step safety proof；
* octave pruning；
* shader data structure；
* 是否能轉換成 density support interval。([Inria HAL][1])

## Volume primitives

### 必讀 B

**Don’t Splat Your Gaussians: Volumetric Ray-Traced Primitives**

要求 agent 回答：

* kernel transmittance closed form；
* free-flight sampling；
* BVH traversal；
* Gaussian 與 Epanechnikov 比較；
* 如何定義 compact interval representation；
* 哪些思想可遷移到 procedural cells。

該工作明確提出 kernel-based scattering/emissive media，並提供不同 kernel 的 closed-form transmittance 與 free-flight sampling。([arXiv][5])

## Neural multiple scattering

### 必讀 C

**A Real-Time Rendering Method for High-Albedo Anisotropic Materials with Multiple Scattering**

要求 agent 回答：

* target 是 radiance 還是 scattering distribution；
* density/transmittance/phase features；
* supervision；
* CNN template；
* runtime；
* 哪些 feature 可壓縮成 interval statistics。([arXiv][4])

### 必讀 D

**Deep Scattering: Rendering Atmospheric Clouds with Radiance-Predicting Neural Networks**

雖然是 2017，但仍是 neural cloud scattering 的核心前置文獻。要求分析：

* hierarchical stencil；
* radiance prediction；
* Monte Carlo coupling；
* rotational/light-direction encoding；
* 雲特有的多重散射監督。([Disney Research Studios][6])

### 必讀 E

**Environmental Volumetric Neural Shading of Clouds for Real-Time Rendering**

要求分析：

* 它使用什麼 asset representation；
* per-pixel thickness；
* sky illumination encoding；
* inference 次數；
* 3.7 ms 是什麼解析度和硬體；
* 18.13 到 22.32 dB ablation 的實際內容；
* 是否能泛化 procedural topology。

該 2026 工作宣稱透過 neural cloud shading 與 rasterization-based pipeline 實現最低約 3.7 ms，並加入 per-pixel thickness 與 sky illumination model。([ACM Digital Library][7])

## Shader neural deployment

### 必讀 F

**RTX Neural Shaders / Cooperative Vectors**

要求 agent 建立：

* FP16 MLP HLSL prototype；
* cooperative-vector prototype；
* ordinary ALU fallback；
* 16/32/64 width benchmark；
* register/occupancy report。

NVIDIA 的工具鏈已提供 tiny neural networks 直接部署至 programmable shader 的路徑；但跨 GPU portability 仍需要普通 shader fallback。([NVIDIA Developer][8])

---

# 二十、實作順序

## Phase H0：重新建立強 baseline

產物：

* production-quality raymarch；
* 720p internal → 4K；
* profiler；
* reference images；
* 積雨雲壓力場景。

Gate：

* baseline 本身已穩定；
* 不可再以舊版 raymarch 當對手。

## Phase H1：A3-Lite interval extraction

產物：

* interval stream debug view；
* empty/analytic/march 類別圖；
* active-length heatmap；
* evaluation-count heatmap。

Gate：

* zero missed cloud support；
* grazing stable；
* large sparse scene density evaluations 至少下降 2 倍。

## Phase H2：B0/B1/B2

產物：

* interval statistic builder；
* transmittance closure；
* cumulant gate；
* analytic fallback。

Gate：

* interior transmittance p95 relative error < 2%；
* boundary 自動退回 raymarch；
* B2 明顯優於 B1。

## Phase H3：Hybrid short raymarch

產物：

* interval-local sampling；
* adaptive budget；
* temporal hysteresis；
* edge reconstruction。

Gate：

* 視覺品質接近 production raymarch；
* 大型積雨雲至少快 1.5 倍。

## Phase H4：MLP dataset

產物：

* path-traced/SOS records；
* feature normalization；
* train/validation/OOD split；
* deterministic dataset versioning。

Gate：

* 不依賴 scene ID；
* dataset 可重現；
* GT noise 低於模型目標誤差。

## Phase H5：Tiny neural residual

產物：

* 24-32-32-4 network；
* PyTorch trainer；
* FP16 exporter；
* HLSL generated weights；
* confidence fallback。

Gate：

* 明顯改善 two-stream/cheap-MS baseline；
* MLP cost < 1.5 ms；
* OOD 不產生嚴重 energy artifact。

## Phase H6：4K optimization

產物：

* 720p→4K temporal reconstruction；
* occupancy-based renderer switching；
* RTX 4070 benchmark；
* p95 frame report。

最終 Gate：

[
3840\times2160,\quad
T_{\mathrm{frame,p95}}\leq33.3\text{ ms}
]

並且至少在主要積雨雲測試中：

[
T_{\mathrm{hybrid}}
\leq
0.67T_{\mathrm{productionRM}}
]

或相同時間下品質有顯著提升。

---

# 最關鍵的決策

我建議現在暫時不要直接推進原本：

> B0 → B1 → B2 → free-path preservation → multiple scattering

的線性路線。

改成兩條並行軌：

```text
Archive v1
   │
   ├── Rendering Track
   │     A3-Lite
   │       → interval extraction
   │       → short raymarch hybrid
   │       → 4K performance gate
   │
   └── Transport Track
         B0/B1/B2
           → interval statistics
           → two-stream baseline
           → tiny neural residual
           → free-path preservation
```

兩條線在 `CloudInterval` 介面合流。

這樣即使最後 B2 或 MLP 的效果有限，**A3-Lite + interval-local raymarch 本身仍是一個完整、可測、可發表或可應用的成果**；而不是必須等整套理論完成後，才知道 renderer 是否實際有競爭力。

[1]: https://inria.hal.science/hal-05046040/file/Fast%20sphere%20tracing%20of%20procedural%20volumetric%20noise%20for%20very%20large%20and%20detailed%20scenes.pdf?utm_source=chatgpt.com "Fast sphere tracing of procedural volumetric noise for very ..."
[2]: https://pascalguehl.github.io/siggraph2025-wave-noise/?utm_source=chatgpt.com "Multi-Dimensional Procedural Wave Noise"
[3]: https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors/?utm_source=chatgpt.com "Neural Rendering in NVIDIA OptiX Using Cooperative Vectors | NVIDIA Technical Blog"
[4]: https://arxiv.org/abs/2401.14051?utm_source=chatgpt.com "A real-time rendering method for high albedo anisotropic materials with multiple scattering"
[5]: https://arxiv.org/abs/2405.15425?utm_source=chatgpt.com "Don't Splat your Gaussians: Volumetric Ray-Traced Primitives for Modeling and Rendering Scattering and Emissive Media"
[6]: https://studios.disneyresearch.com/2017/11/20/deep-scattering-rendering-atmospheric-clouds-with-radiance-predicting-neural-networks/?utm_source=chatgpt.com "Rendering Atmospheric Clouds with Radiance-Predicting ..."
[7]: https://dl.acm.org/doi/10.1145/3820020?utm_source=chatgpt.com "Environmental Volumetric Neural Shading of Clouds for ..."
[8]: https://developer.nvidia.com/blog/nvidia-releases-rtx-neural-rendering-tech-for-unreal-engine-developers/?utm_source=chatgpt.com "NVIDIA Releases RTX Neural Rendering Tech for Unreal Engine Developers | NVIDIA Technical Blog"
