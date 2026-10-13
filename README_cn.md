# 🔥 RSS：稀疏自编码器中的表示重分配

**Ziye Mei, Haotian Ma**<br>
University of Wisconsin–Madison

🌍 [English](README.md) | [中文](README_cn.md)

👉 一句话。按设计不改变请求本身的 prompt 改写，仍会让稀疏自编码器 SAE 的特征重新分配激活。我们提出重分配敏感度分数 RSS，它选出变化大、反复出现且方向一致的特征响应，并在没有参与选择的变体上检验这些特征。在来自 8 个语言模型的 27 个预训练 SAE 上，所测试的改写给出一致可复现的重分配规律。在两个冻结的安全读出上，把 RSS 支撑集的侵蚀与干净 margin 结合，预测反复出错的 AUROC 为 0.935 和 0.958，所用支撑集中位数为 20 和 19 个特征，用全部特征的侵蚀则为 0.965 和 0.964。

<p align="center">
  <b>📄 arXiv 论文即将上线</b> &nbsp; | &nbsp;
  <a href="https://uniquemax2005.github.io/rss-redistribution/"><b>🎞️ 5 分钟讲解</b></a> &nbsp; | &nbsp;
  <a href="#-demo"><b>🧪 Demo</b></a>
</p>

## 🎥 概览

<p align="center">
  <img src="assets/figures/fig1.png" alt="Figure 1" width="900"/>
</p>

**Figure 1.** RSS under prompt transformations. In the illustrated five-variant selection, boxed features change in the
same direction in at least four variants. MeanRSS averages their absolute response magnitudes.

---

## 🎞️ 5 分钟讲解

<p align="center">
  <a href="https://uniquemax2005.github.io/rss-redistribution/"><img src="assets/tour_preview.png" alt="5 分钟讲解" width="900"/></a>
</p>

💻 点击预览图即可在浏览器中打开讲解，地址是 <https://uniquemax2005.github.io/rss-redistribution/>。用方向键翻页，按 N 看讲稿，按 D 看每页的细节。同一个页面也是仓库里的独立文件 [`docs/index.html`](docs/index.html)，离线也能打开。

---

## ✨ 亮点

- 🔁 **规律可复现。** 在全部 27 个 SAE 上，加上下文和套格式这两类改写都比局部改动让表示变化更大。归一化后两组相差 0.874，1,000 个 prompt 中每一个都是上下文和包装一组更大。
- 🎯 **规则简单。** RSS 保留变化最大的特征，要求它在 5 个变体中至少 4 个有响应，并且至少 80% 的响应方向相同。
- 🧭 **方向能保持。** 在留出变体上，RSS 特征沿选定方向响应的比例为 0.952，同样数量的随机特征为 0.758。
- 🧮 **变化不等于出错。** 对固定的线性读出，选中特征的贡献可以精确拆成读出权重、响应大小和与选定方向的一致程度。决策会不会翻转还取决于干净 margin。
- 🛡️ **小而有用。** RSS 选特征时不看读出，但选出的 20 个和 19 个特征就能重建留出变体上的 logit 变化，$`R^2`$ 为 0.58 和 0.63，并保留完整读出的大部分失败排序能力。
- ⚡ **代码很小。** 选择器就是一个 NumPy 函数 [`rss.py`](rss.py)。GPT-2 demo 在 CPU 上约一分钟跑完。

---

## 💡 为什么用 RSS？

稀疏自编码器 SAE 把语言模型的稠密激活分解成稀疏特征。相关的输入会改变哪些特征响应，加上下文、换格式或换词这类 prompt 改写也会改变模型行为。要用 SAE 特征描述这种变化，就需要把只属于某一个输入的响应和在相关输入之间反复出现的规律区分开。

只盯一个特征，能看到它的响应，却会漏掉别处的变化。对整体激活算距离，能概括变化有多大，却说不出是哪些特征在承载这次变化。我们把激活在稀疏基上分布方式的改变称为表示重分配。RSS 选出变化大、并且朝同一方向反复出现的特征，在一组变体上确定这些特征，再到另一组不相交的变体上检验。

反复出现的变化不一定会影响下游。选中的特征可能读出权重很小，不同特征的贡献可能相互抵消，读出分数的变化也不一定越过决策阈值。所以我们单独测量这些变化对固定线性读出的贡献，并和干净输入上剩下的 margin 对比。

---

## 📐 RSS 测量什么

🧩 **特征变化。** 所有量都针对一个预训练 SAE、一层、一个基础 prompt 和一种改写设置来定义。记 $`x_0`$ 为原始 prompt，$`x_v`$ 为它的改写变体，$`v\in V`$。在对齐后的目标位置，$`a_f(x)`$ 是特征 $`f`$ 的激活。它的变化和平均响应幅度为

$$\delta_{v,f}=a_f(x_v)-a_f(x_0),\qquad h_f=\frac{1}{\lvert V\rvert}\sum_{v\in V}\lvert\delta_{v,f}\rvert .$$

$`h_f`$ 对绝对值求平均，所以方向相反的变化不会互相抵消。

🔁 **反复出现与方向。** 记 $`n_f^{+}`$ 为 $`n`$ 个选择观测中大于 $`\tau`$ 的个数，$`n_f^{-}`$ 为小于 $`-\tau`$ 的个数，则

$$r_f=\frac{n_f^{+}+n_f^{-}}{n},\qquad c_f=\frac{\max(n_f^{+},n_f^{-})}{n_f^{+}+n_f^{-}},\qquad s_f=\mathrm{sign}(n_f^{+}-n_f^{-}).$$

$`r_f`$ 衡量反复出现的程度，$`c_f`$ 衡量方向一致性，$`s_f`$ 是主导方向。

🎯 **RSS 支撑集与 MeanRSS。** 在 $`h_f`$ 最大的 $`M`$ 个特征中，满足 $`r_f\ge\rho`$ 且 $`c_f\ge\kappa`$ 的特征为合格特征。RSS 保留其中 $`h_f`$ 最大的 $`k=\min(K,\lvert\mathcal E_M\rvert)`$ 个，作为响应支撑集 $`S`$，并定义

$$\mathrm{MeanRSS}=\frac{1}{\lvert S\rvert}\sum_{f\in S}h_f .$$

我们取 $`\rho=\kappa=0.8`$，$`\tau=10^{-5}`$，$`M=400`$，$`K=20`$。在 $`n=5`$ 个观测下，每个选中特征至少 4 次沿主导方向变化，至多 1 次反向。

📏 **归一化 MeanRSS 与留出对比。** 为了比较尺度不同的 SAE，对每种设置 $`g`$ 先在 prompt $`i`$ 上平均 MeanRSS，再除以六种设置的平均

$$R_g=\mathrm{mean}_i\,\mathrm{MeanRSS}_{i,g},\qquad \widetilde R_g=\frac{R_g}{\frac{1}{6}\sum_{g'}R_{g'}} .$$

这样每个 SAE 的六个值平均为 1。对一个特征集合 $`A`$，$`H_g(A)`$ 是 $`\lvert\delta_{v,f}\rvert`$ 在 prompt、$`A`$ 中特征和留出变体上的平均，$`\widetilde H_g(A)`$ 是同样归一化后的值。记 $`\mathcal C`$ 为四种上下文和包装设置，$`\mathcal L`$ 为两种局部改动，留出对比为

$$\Delta(A)=\frac{1}{4}\sum_{g\in\mathcal C}\widetilde H_g(A)-\frac{1}{2}\sum_{g\in\mathcal L}\widetilde H_g(A).$$

🛡️ **Margin 与侵蚀。** 冻结的线性探针读取标准化激活 $`\widetilde{\mathbf a}(x)`$，分数为 $`z(x)=\mathbf w^\top\widetilde{\mathbf a}(x)+b`$，阈值为 $`t`$。对类别符号 $`y\in\lbrace -1,+1\rbrace`$，margin 为 $`m(x)=y\,[z(x)-t]`$，正的 margin 表示判对，$`m_0=m(x_0)`$ 是干净 margin。在变体集合 $`U`$ 上，特征 $`f`$ 对 margin 的侵蚀为

$$e_f^U=-y\,w_f\,\frac{1}{\lvert U\rvert}\sum_{v\in U}\widetilde\delta_{v,f}.$$

对选中特征，它可以精确分解为 $`e_f^U=-y\,w_f\,s_f\,A_f^U\,\eta_f^U`$。其中 $`A_f^U`$ 是标准化响应的平均绝对值，$`\eta_f^U\in[-1,1]`$ 是它与选定方向的一致程度。在选择变体上，全部侵蚀和支撑集承担的部分分别为 $`E=\sum_f e_f`$ 和 $`E_S=\sum_{f\in S}e_f`$。

🧾 **剩余 margin 与贡献保留。** 从干净 margin 中减去侵蚀，得到剩余 margin $`m_{\mathrm{res}}=m_0-E`$ 和 $`m_{\mathrm{res}}^{S}=m_0-E_S`$。完整的剩余 margin 等于选择变体上改写后 margin 的平均。在留出变体上，margin 贡献保留率 MCR 是支撑集保留的绝对贡献所占比例

$$\mathrm{MCR}=\frac{\sum_{f\in S}\lvert e_f^{\mathrm{ho}}\rvert}{\sum_f\lvert e_f^{\mathrm{ho}}\rvert}.$$

⚠️ **失败、AUROC 与审查。** 对每个测试 prompt 和设置，每个算子取一个实现组成 5 个选择变体，另外两个实现组成 10 个留出变体。反复出错指 10 个留出 margin 中至少 2 个为负。对风险分数 $`q`$，AUROC 是一个出错的样本 $`i`$ 排在一个没出错的样本 $`j`$ 前面的概率

$$\mathrm{AUROC}=\Pr\big(q(i)>q(j)\big).$$

选择性审查按剩余 margin 给样本排序，报告风险最高的 20% 中包含了多少留出错误。

🔧 **重建与恢复。** 对一个留出变体，把 logit 变化 $`d=z(x_v)-z(x_0)`$ 和支撑集之和 $`\widehat d_S=\sum_{f\in S}w_f\widetilde\delta_{v,f}`$ 比较，用决定系数 $`R^2`$ 衡量。把选中特征的读出输入恢复成干净值，得到

$$m_{\mathrm{cf}}(x_v)=m_0+y\sum_{f\notin S}w_f\widetilde\delta_{v,f}.$$

改写后判错且 $`m_{\mathrm{cf}}(x_v)>0`$ 的样本就被纠正了。这个反事实只改读出的计算，不改语言模型的激活。

---

## 🔎 方法对比

🧭 **哪些特征承载改写带来的差异。** 把 RSS 和三种不看改写的特征集合对比。Clean fixed 取从原始 prompt 上选出的特征。Global fixed 取在整个原始语料上最活跃的特征。Dictionary mean 对全部 SAE 特征等权平均。每种集合都在留出变体上用 $`\Delta(A)`$ 评估，数值越大说明区分越清楚。表中数字是 Figure 3 里 27 个 SAE 的平均。

| 特征集合 | 留出对比 $`\Delta(A)`$ | RSS 更大的 SAE 数 |
|---|:---:|:---:|
| 🏆 **RSS** | ✅ **0.504** | |
| Clean fixed | 0.376 | ✅ 27 个中 24 个 |
| Global fixed | 0.365 | ✅ 27 个中 26 个 |
| Dictionary mean | ❌ 0.306 | ✅ 27 个中 24 个 |

🎯 **选定方向能否保持。** 变化大的特征也许本来就稳定，不需要反复出现和方向一致的约束。所以把 RSS 和只按幅度选择对比，两者用相同的候选池和支撑集大小。表中是两个安全探针上留出变体沿选定方向响应的比例。

| 选择方式 | Gemma 3 4B | Llama 3.1 8B |
|---|:---:|:---:|
| 🏆 **RSS** | ✅ **0.876** | ✅ **0.871** |
| 只按幅度 | 0.816 | 0.794 |

🛡️ **哪个分数能排出读出失败。** 所有分数都在 5 个选择变体上计算，用 10 个留出变体上反复出错的 AUROC 来评判。MeanRSS 只用支撑集的响应大小，第二行再按探针系数的绝对值加权。干净 margin 只用探针在原始 prompt 上的结果。其余几行用上面定义的侵蚀量。

| 分数 | 特征 | Gemma 3 4B | Llama 3.1 8B |
|---|:---:|:---:|:---:|
| MeanRSS | RSS 支撑集 | ❌ 0.526 | ❌ 0.506 |
| 按探针权重加权的 MeanRSS | RSS 支撑集 | ❌ 0.676 | ❌ 0.581 |
| RSS 侵蚀 $`E_S`$ | RSS 支撑集 | 0.753 | 0.735 |
| 全部侵蚀 $`E`$ | 全部 | 0.785 | 0.705 |
| 干净 margin $`m_0`$ | 全部 | 0.863 | 0.901 |
| 🏆 **RSS 剩余 margin** $`m_{\mathrm{res}}^{S}`$ | ✅ **20 和 19** | ✅ **0.935** | ✅ **0.958** |
| 完整剩余 margin $`m_{\mathrm{res}}`$ | 415 和 862 | 0.965 | 0.964 |

🔬 **哪些特征能定位读出的变化。** 各支撑集大小都与 RSS 支撑集相同。Mismatched setting 用为另一种改写选出的 RSS 支撑集。Global instability 用在语料上变化最大的特征，random 用随机特征。$`R^2`$ 衡量支撑集重建留出 logit 变化的程度，纠正错误是恢复支撑集后被纠正的改写错误所占比例。

| 支撑集 | Gemma $`R^2`$ | Llama $`R^2`$ | Gemma 纠正错误 | Llama 纠正错误 |
|---|:---:|:---:|:---:|:---:|
| 🏆 **RSS** | ✅ **0.579** | ✅ **0.627** | ✅ **72.2%** | ✅ **71.8%** |
| Mismatched setting | 0.438 | 0.490 | 60.3% | 61.1% |
| Global instability | ❌ 0.017 | ❌ 0.009 | ❌ 6.7% | ❌ 2.3% |
| Random | ❌ 0.014 | ❌ 0.017 | ❌ 1.2% | ❌ 1.5% |

---

## ⚙️ 安装

```bash
git clone https://github.com/UniqueMax2005/rss-redistribution.git
cd rss-redistribution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`rss.py` 只依赖 NumPy，只用选择器时 `pip install numpy` 就够了。demo 另需 PyTorch、TransformerLens、SAE Lens 和 Matplotlib。下面的 demo 输出由 Python 3.12 和 [`requirements-lock.txt`](requirements-lock.txt) 里的版本得到，`pip install -r requirements-lock.txt` 会装上完全相同的版本。

✅ **测试。** `python -m pytest` 检查论文依赖的性质，包括严格阈值、空支撑集记为不可用、Proposition 1 和 2、对尺度以及特征和变体顺序的不变性、输入检查，以及两份 README 快速上手的输出。每次 push 都由 GitHub Actions 自动运行。

---

## 🚀 快速上手

```python
import numpy as np
from rss import rss

# 5 个变体的 SAE 激活减去原始 prompt 的激活，每行一个变体，每列一个特征
delta = np.array([[2.0, 0.0, -1.0,  0.3],
                  [1.8, 0.0, -1.2, -0.3],
                  [2.2, 0.5, -0.9,  0.3],
                  [1.9, 0.0, -1.1, -0.3],
                  [0.0, 0.0, -1.0,  0.3]])

support, direction, mean_rss = rss(delta)   # rho = kappa = 0.8, tau = 1e-5, M = 400, K = 20
print(support, direction, round(mean_rss, 2))
# [0 2] [ 1 -1] 1.31
```

💡 特征 0 在 5 个变体中有 4 个上升，特征 2 在 5 个中全部下降，所以都保留。特征 3 每次都变，但方向来回翻，所以被去掉。

👉 在 GPT-2 Small 上运行

```bash
python demo.py
```

🧰 **用在自己的 SAE 上。** 为每个原始 prompt 写若干变体，保持请求不变，并让目标短语恰好出现一次。在与目标短语重叠的最后一个 token 处读取 SAE，做法同 [`demo.py`](demo.py) 里的 `sae_features`。在查看任何特征之前先把变体分成选择变体和留出变体，在前一组上选择，在后一组上检验选定方向。

```python
from rss import rss, held_out_rate

# a0 的形状是 (n_features,)，a_sel 和 a_ho 的形状是 (n_variants, n_features)
support, direction, mean_rss = rss(a_sel - a0)
rate = held_out_rate(a_ho - a0, support, direction)   # 每个选中特征一个值
```

`held_out_rate` 就是论文 Equation 20 的留出响应率，也就是 Figure 4A 里的量。两个函数遇到 nan 或 inf、形状不对或参数超出范围时都会报错，不会返回一个看似正常的数字。

---

## 📊 主要结果

#### 🔁 Figure 2. Normalized MeanRSS across six prompt transformations and 27 pretrained SAEs

27 个 SAE 等权平均的归一化 MeanRSS $`\widetilde R_g`$。

| 无关上下文 | 干扰上下文 | 元语言包装 | 格式包装 | 标点噪声 | 同类词替换 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1.466** | **1.341** | **1.199** | **1.159** | 0.525 | 0.309 |

在全部 27 个 SAE 上，上下文和包装的平均都高于局部改动，平均差 0.874，95% 区间为 0.867 到 0.881。1,000 个 prompt 中每一个都是如此，其中 997 个 prompt 的每一种上下文或包装设置都高于两种局部改动。在四个强度等级上，强度与 MeanRSS 的 prompt 级 Spearman 相关的中位数，标点改动为 0.95，无关上下文为 0.74，格式包装为 0.50。

#### 🧭 Figure 3. Response contrast on variants excluded from feature selection

在 27 个 SAE 中，RSS 区分上下文包装和局部改动的能力强于每一种对比方法的有 24 到 26 个，三者中最强的有 22 个。在留出变体上，RSS 支撑集沿选定方向响应的比例为 0.952，候选池和大小相同的随机支撑集为 0.758。

#### 🎯 Figure 4. Feature validation on held out variants

| | Gemma 3 4B | Llama 3.1 8B |
|---|:---:|:---:|
| 沿选定方向的响应比例，RSS 和只按幅度 | **0.876** 和 0.816 | **0.871** 和 0.794 |
| 平均 $`\lvert e_f^{\mathrm{ho}}\rvert`$，RSS 和按权重匹配的特征 | **0.240** 和 0.022 | **0.370** 和 0.022 |
| RSS 贡献更大的匹配对比例 | 96.9% | 98.0% |
| $`K=20`$ 时的 MCR，RSS 和 mismatched RSS | **43.0%** 和 32.6% | **56.2%** 和 43.5% |

分解给出的符号规则与留出贡献的符号一致的比例为 97.4% 和 97.1%。RSS 并不追求最大的贡献保留。在选择变体上按 $`\lvert e_f\rvert`$ 排序要用到读出，在两个模型上都能保留 90% 以上。

#### ⚖️ Figure 5. Readout change and recurrent readout failure on held out variants

在每一种改写下，探针 AUROC 都保持在 0.90 以上，但仍有 4.2% 到 9.5% 的决策发生改变。在选择变体上估计的全部侵蚀与留出侵蚀的相关为 Gemma 0.923、Llama 0.896，RSS 支撑集的侵蚀为 0.737 和 0.793。只用干净 margin，反复出错的 AUROC 为 0.863 和 0.901。从干净 margin 中减去 RSS 支撑集的侵蚀，得到 0.935 和 0.958，完整剩余 margin 为 0.965 和 0.964。这些都是用已知类别标签做的离线排序。

#### 🔬 Figure 6. Utility of RSS supports selected for each prompt and transformation within the frozen readout

| | 完整读出 | RSS 支撑集 |
|---|:---:|:---:|
| 特征数，Gemma 3 4B 和 Llama 3.1 8B | 415 和 862 | **20 和 19** |
| 侵蚀的 Spearman $`\rho`$ | 0.923 和 0.896 | 0.737 和 0.793 |
| 剩余 margin 的 AUROC | 0.965 和 0.964 | 0.935 和 0.958 |
| 审查 20% 时捕获的错误 | 94.1% 和 91.9% | 89.2% 和 92.8% |
| 审查 20% 时捕获的有害漏检 | 90.9% 和 93.5% | 87.3% 和 94.4% |

RSS 重建留出 logit 变化的 $`R^2`$ 约为 0.58 和 0.63。把选中特征的读出输入恢复成干净值，可以纠正 72.2% 和 71.8% 的改写错误。这里紧凑的是每个样本用到的那组变化，而不是一个替代完整干净计算的缩小版探针。

---

## 🧪 Demo

`demo.py` 用 GPT-2 Small 和它的公开 SAE，在我们写的 24 个 prompt 上做 6 种改写，每种 5 个变体。它在每个变体都保留的那部分 prompt 的最后一个 token 处读取 SAE，对每个 prompt 和设置运行 `rss`，报告 MeanRSS、归一化值，以及上下文包装与局部改动之间的差。`python demo.py` 默认用第 8 层，CPU 上约一分钟。用 `--layer` 可以选论文里 GPT-2 的另外三层 4、6、10。

```text
GPT-2 Small, layer 8, 24 prompts

Transformation            MeanRSS  normalized
Irrelevant context          2.121       1.107
Distractor context          2.481       1.295
Metalinguistic wrapper      1.977       1.032
Format wrapper              2.341       1.222
Punctuation noise           1.231       0.643
Typed substitution          1.344       0.702

Context and wrappers minus local edits: 0.492
Prompts where context and wrappers move more: 24 / 24
```

<p align="center">
  <img src="assets/demo_layer8.png" alt="Demo 输出" width="850"/>
</p>

✅ demo 和论文的规律一致。上下文和包装比局部改动让表示变化更大，同类词替换动到的是另一批特征。第 4、6、8、10 层的差分别为 0.903、0.717、0.492、0.390，24 个 prompt 中有 23 或 24 个方向一致。论文用 1,000 个生成的 prompt，这四层的差为 0.939 到 0.979。

---

## 🗺️ 图表索引

图注取自论文原文。

| 图 | 图注 | Demo 能复现 |
|---|---|:---:|
| [Figure&nbsp;1](assets/figures/fig1.png) | RSS under prompt transformations. In the illustrated five-variant selection, boxed features change in the same direction in at least four variants. MeanRSS averages their absolute response magnitudes. | |
| [Figure&nbsp;2](assets/figures/fig2.png) | Normalized MeanRSS across six prompt transformations and 27 pretrained SAEs. Each SAE's six setting means are divided by their average, so the plot compares transformation profiles rather than absolute activation scales. Large outlined diamonds show averages across SAEs. | ✅ 子图 A |
| [Figure&nbsp;3](assets/figures/fig3.png) | Response contrast on variants excluded from feature selection. Each point is the normalized context and wrapper mean minus the local edit mean for the indicated feature set and SAE layer. Clean fixed: features selected from clean prompts. Global fixed: features selected by activity across the clean corpus. Dictionary mean: equal average over all SAE features. | |
| [Figure&nbsp;4](assets/figures/fig4.png) | Feature validation on held out variants. A, response rate in the direction chosen on selection variants, comparing RSS with magnitude alone. B, mean $`\lvert e_f^{\mathrm{ho}}\rvert`$ for RSS features and unselected features matched by readout coefficient magnitude. C–D, MCR as the support cap $`K`$ varies. The $`\lvert e_f\rvert`$ ranking uses contributions from the selection variants. | |
| [Figure&nbsp;5](assets/figures/fig5.png) | Readout change and recurrent readout failure on held out variants. A, Spearman correlation with held out margin erosion and AUROC for recurrent failure. Correlation measures erosion transfer, while AUROC measures failure ranking; residual margin correlations are oriented toward increasing risk. B, clean margin versus full residual margin for Gemma 3 4B, colored by held out recurrent failure. The dashed line marks zero average selection margin. | |
| [Figure&nbsp;6](assets/figures/fig6.png) | Utility of RSS supports selected for each prompt and transformation within the frozen readout. A, $`R^2`$ for reconstructed held out logit changes. B, readout errors reversed by restoring selected feature values. C–D, errors and harmful false negatives captured at a 20% offline review budget using known labels. Error bars show 95% bootstrap confidence intervals. | |
| [Figure&nbsp;S1](assets/figures/figS1.png) | Normalized MeanRSS profiles for the six transformations across all sampled layers of the eight models. Normalization follows Figure 2. | ✅ `--layer` |
| [Figure&nbsp;S2](assets/figures/figS2.png) | Overlap between the feature supports selected under different transformations, averaged over all units and 1,000 prompts. A, pairwise Jaccard overlap between transformations; B, averages within each model. IC denotes irrelevant context, DC distractor context, MW metalinguistic wrapper, FW format wrapper, PN punctuation noise, and TL typed lexical substitution. | ✅ 子图 B |
| [Figure&nbsp;S3](assets/figures/figS3.png) | Normalized MeanRSS by prompt topic. A, topic level means for each transformation; B, the ratio of the context and wrapper mean to the local edit mean for each topic. Transformation abbreviations as in Figure S2. | |
| [Figure&nbsp;S4](assets/figures/figS4.png) | Profile of the six transformations after removing one model at a time, compared with the profile over all units. Transformation abbreviations as in Figure S2. | |
| [Figure&nbsp;S5](assets/figures/figS5.png) | Comparison of the main feature selection rule with an alternative rule that treats all fifteen variants individually. A, mean support size; B, rates of undersized and empty selections; C, Jaccard overlap between the supports chosen by the two rules; D, unit level normalized MeanRSS under the two rules. Transformation abbreviations as in Figure S2. | |
| [Figure&nbsp;S6](assets/figures/figS6.png) | Probe AUROC for the two safety readouts on untransformed test prompts and under each transformation. Transformation abbreviations as in Figure S2. Error bars show 95% bootstrap confidence intervals. | |
| [Figure&nbsp;S7](assets/figures/figS7.png) | Normalized MeanRSS and margin erosion on held out variants for each transformation, evaluated on the audited safety units. Transformation abbreviations as in Figure S2. Error bars show 95% bootstrap confidence intervals. | |
| [Figure&nbsp;S8](assets/figures/figS8.png) | Effect of transformation strength on the audited safety units. A, margin erosion under three transformations rebuilt at four increasing intensity levels, with 95% bootstrap confidence intervals. B, MeanRSS for the same series, relative to the weakest level. | |
| [Figure&nbsp;S9](assets/figures/figS9.png) | Augmentation guided by MeanRSS compared with uniform sampling. A, differences between the two sampling methods in clean, mean, and worst transformation AUROC, together with the reduction in harmful false negative rate under the worst transformation. Error bars show 95% bootstrap confidence intervals. B, harmful false negative rate averaged over the six transformations across five runs. Bars show the run mean and whiskers the observed range. | |

<details>
<summary>论文使用的 SAE</summary>

取自论文 Table S3。27 个 SAE 都是公开的 checkpoint。

| 模型 | 层 | SAE checkpoint | 宽度 | Hook | 架构与稀疏度 |
|---|---|---|---:|---|---|
| GPT-2 Small | 4, 6, 8, 10 | `GPT2-Small-SAEs-Reformatted` | 24,576 | `resid_pre` | ReLU，32×，L1 系数 $`8\times10^{-5}`$ |
| Pythia-70M | 1, 3, 5 | `pythia-70m-deduped__res-sm` | 32,768 | `resid_post` | ReLU，standard，稀疏度未报告 |
| Gemma 3 270M | 5, 9, 12, 15 | `gemma-scope-2-270m-pt` | 16,384 | `resid_post` | JumpReLU，第 5 层 $`L_0=18`$，其余为 20 |
| Gemma 3 1B | 7, 13, 17, 22 | `gemma-scope-2-1b-pt` | 16,384 | `resid_post` | JumpReLU，第 7 层 $`L_0=18`$，其余为 20 |
| Gemma 3 4B | 9, 17, 29 | `gemma-scope-2-4b-pt` | 65,536 | `resid_post` | JumpReLU，名义 $`L_0`$ 为 53、60、60 |
| Mistral 7B | 8, 16, 24 | `mistral_7b_layer_L` | 65,536 | `resid_pre` | ReLU，16×，实测 $`L_0`$ 为 92、74、75 |
| Llama 3.1 8B | 8, 16, 24 | `Llama3.1-L{8,16,24}R-8x` | 32,768 | `resid_post` | JumpReLU，8×，实测 $`L_0=50`$ |
| Gemma 2 2B | 5, 12, 19 | `gemma-scope-2b-pt-res` `width_65k` | 65,536 | `resid_post` | JumpReLU，实测 $`L_0`$ 为 53、72、63 |

demo 通过 SAE Lens 以 `gpt2-small-res-jb` 加载 GPT-2 Small 的 SAE，位置是 `blocks.L.hook_resid_pre`。

</details>

<details>
<summary>展开全部图</summary>

| | |
|---|---|
| <img src="assets/figures/fig2.png" width="420"/><br>Figure 2 | <img src="assets/figures/fig3.png" width="420"/><br>Figure 3 |
| <img src="assets/figures/fig4.png" width="420"/><br>Figure 4 | <img src="assets/figures/fig5.png" width="420"/><br>Figure 5 |
| <img src="assets/figures/fig6.png" width="420"/><br>Figure 6 | <img src="assets/figures/figS1.png" width="420"/><br>Figure S1 |
| <img src="assets/figures/figS2.png" width="420"/><br>Figure S2 | <img src="assets/figures/figS3.png" width="420"/><br>Figure S3 |
| <img src="assets/figures/figS4.png" width="420"/><br>Figure S4 | <img src="assets/figures/figS5.png" width="420"/><br>Figure S5 |
| <img src="assets/figures/figS6.png" width="420"/><br>Figure S6 | <img src="assets/figures/figS7.png" width="420"/><br>Figure S7 |
| <img src="assets/figures/figS8.png" width="420"/><br>Figure S8 | <img src="assets/figures/figS9.png" width="420"/><br>Figure S9 |

</details>

---

## 🙏 致谢

本项目基于 [SAE Lens](https://github.com/decoderesearch/SAELens)、[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) 和 [Neuronpedia](https://www.neuronpedia.org)，使用了 [GPT-2 Small](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted)、[Pythia-70M](https://huggingface.co/ctigges/pythia-70m-deduped__res-sm_processed)、[Gemma Scope](https://huggingface.co/google/gemma-scope)、[Gemma Scope 2](https://huggingface.co/google/gemma-scope-2-4b-pt)、[Mistral-7B](https://huggingface.co/JoshEngels/Mistral-7B-Residual-Stream-SAEs) 和 [Llama Scope](https://huggingface.co/OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x) 的公开 SAE，以及 [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) 数据。
