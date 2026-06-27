# GR — 每日中文 BERT 自动微调 + 评估(MoE+Multi-LoRA)

> 利用 **GitHub Actions** 每日自动跑一次中文 BERT 的轻量微调与评估。
> 每次运行会从候选池**随机抽取一个未用过的中文数据集**作为新训练素材,
> 配合 **MoE + Multi-LoRA(MoMo)**,让模型在「数据 + 架构」两个维度随时间演化。
> 这是一个实验性项目,目的是**探究与利用 GitHub** 的免费计算能力。

## ✨ 项目目标

[![Daily Fine-tune](https://github.com/Annieif/GR/actions/workflows/daily-finetune.yml/badge.svg)](https://github.com/Annieif/GR/actions/workflows/daily-finetune.yml)
![Daily Fine-tune](https://img.shields.io/badge/daily-fine--tune-blueviolet?style=flat-square)
![Model](https://img.shields.io/badge/model-MacBERT--base-orange?style=flat-square)
![LoRA+MoE](https://img.shields.io/badge/LoRA%2BMoE-yes-success?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

- 每天 0 点(北京时间)自动在 GitHub 云端跑一次模型微调
- 训练后**自动评估** base 与微调后模型,产出 PPL / top-k 准确率
- 支持 **全参微调** / **PEFT LoRA** / **MoE+Multi-LoRA** 三种模式
- 每次运行**自动从 HF Hub 拉取一个未用过的随机中文数据集**(`fetch_random_data.py`)
- 拉取过的数据集与 manifest 一起 commit 回仓库,后续运行避免重复
- 不需要本地 Python 环境
- 模型与训练日志上传为 artifact,可下载、对比、观察逐步演化

## 🤖 选型

| 项目 | 选择 | 理由 |
| --- | --- | --- |
| Base 模型 | **`hfl/chinese-macbert-base`** | 纯 PyTorch,中文原生,~400MB,MIT 协议 |
| 备选 | `bert-base-chinese` / `hfl/chinese-roberta-wwm-ext` | 直接换 `--model_name` |
| 任务 | **Masked Language Modeling** | 无标注、增量训练友好 |
| 数据 | `data/corpus.txt` + `data/extra_corpora/*.txt` | 基础语料 + 每日随机追加 |
| 路由 | top-2,默认 4 expert | 平衡性能与训练成本 |
| Runner | `ubuntu-latest`(免费,2 核 7G) | 加缓存后单次训练 < 15 分钟 |

## 📂 仓库结构

```
GR/
├── .github/workflows/daily-finetune.yml   # 每日 cron 工作流
├── data/
│   ├── corpus.txt                         # 基础训练语料
│   ├── eval.txt                           # 评估语料
│   ├── dataset_pool.json                  # 候选中文数据集池
│   ├── used_datasets.json                 # 已用过的数据集 manifest
│   ├── extra_corpora/                     # 每天自动追加的新语料
│   └── fetch_random_data.py               # 随机拉取脚本
├── train.py                               # 训练(支持全参/LoRA/MoMo)
├── evaluate.py                            # 评估(PPL + top-k + fill-mask)
├── momo_lora.py                           # MoE + Multi-LoRA 注入模块
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 使用方式

> 🎯 **默认 = 完整体验**:`use_momo=true`(MoE+Multi-LoRA) · `fetch_random_data=true`(自动拉新数据集) · `skip_base_eval=false`(base vs ft 对比) · `push_to_hub=false`(需 secret 才推送)
> 点 Run workflow 不改任何参数,就能跑出完整 pipeline。

### 1. 直接使用
- Fork / push 到 GitHub,启用 Actions
- 每天 UTC 02:00 自动跑一次(北京时间 10:00)
- 也可在 Actions 页面手动触发并改参数

### 2. 手动触发参数(关键)

> **默认就是「完整体验」**:MoE+Multi-LoRA ✅ · 自动拉新数据集 ✅ · 自动评估 ✅ · 日志 commit ✅
> **HF Hub 推送默认关闭**(需要 secret token,在 `push_to_hub=true` 时才生效)

| 输入 | 默认 | 说明 |
| --- | --- | --- |
| `model_name` | `hfl/chinese-macbert-base` | 任何纯 PyTorch 中文模型 |
| `epochs` | `3` | 训练轮数 |
| `batch_size` | `16` | 每设备 batch size |
| `fetch_random_data` | **`true`** | 本次是否拉取一个未用过的随机数据集(自动扩语料) |
| `max_random_samples` | **`500`** | 拉取数据集最多取多少条 |
| `use_lora` | `false` | 启用 PEFT LoRA(与 `use_momo` 互斥) |
| `lora_r` / `lora_alpha` / `lora_dropout` | 8 / 16 / 0.1 | PEFT LoRA 超参 |
| `merge_lora` | `true` | 训练后把 LoRA 合并回 base |
| **`use_momo`** | **`true`** | **启用 MoE + Multi-LoRA(完整体验核心)** |
| `momo_n_experts` | 4 | 专家数 |
| `momo_top_k` | 2 | top-k 路由 |
| `momo_lora_r` | 8 | 单个 LoRA expert 的秩 |
| `momo_lora_alpha` | 16 | LoRA alpha |
| `momo_target` | `query,value` | 注入目标 Linear 名称 |
| `momo_aux_alpha` | 0.01 | 负载均衡 loss 系数 |
| `skip_base_eval` | `false` | 跳过 base 评估(加速) |
| `push_to_hub` | **`false`** | 推送到 HF Hub(需 HF_TOKEN secret) |
| `hub_repo_id` | 空 | HF Hub 仓库名 |
| `use_previous_model` | **`true`** | 从上一个 Release 下载上次微调后的完整模型作为本次起点(实现跨日累积微调) |
| `auto_release` | **`true`** | 本次跑完后自动创建 GitHub Release(包含 `model.zip` + 评估/训练日志) |
| `release_zip_name` | `model.zip` | Release 附件 zip 文件名 |

### 3. 三种训练模式

**全参微调**(默认,~400MB checkpoint):
- 训练所有参数
- 适合:有充足 GPU / 想看完整范式

**PEFT LoRA**(`--use_lora`,~10MB adapter):
- 单一 LoRA adapter
- 默认 `--merge_lora`,合并后保存为完整模型,evaluate.py 不用感知 LoRA

**MoE + Multi-LoRA**(`--use_momo`,~15MB adapter) — **本项目新功能**:
- 在 BERT 的 query / value(可配置)位置注入 N 个 **LoRA expert**
- 配合一个 **router(gating network)**:每个 token 由 top-k 个 expert 共同处理
- 训练时同步加 **load-balancing auxiliary loss**,避免 router 坍缩
- 实现见 `momo_lora.py`(`MoLoRALinear` / `inject_momo_lora` / `add_momo_aux_loss_hook`)
- 适合:想学**路由/专家分工**、想保留多组可单独关掉的适配器

#### MoMo 怎么工作

对一个 `nn.Linear` 注入:
```
y = base(x)                               ← frozen
  + Σ_k  w_k(i,k) * LoRA_i(x)             ← N experts, top-k 路由
  + α·N·Σ_i  f_i·P_i  (aux loss)          ← 负载均衡
```
- `base` 冻结,只训 expert (`lora_A/B`) 和 `router`
- 默认只注入 `query` + `value`(Q/V 是 LoRA 性价比最高的位置)
- 12 层 × 2 (Q,V) = 24 个 `MoLoRALinear`,共 24×4=96 个 LoRA expert
- 每个 expert ~25KB,总计 ~2.4MB 可训练参数

### 4. 每日自动拉取数据集

每次 workflow 会:
1. 读 `data/used_datasets.json` 拿「已用集」
2. 从 `data/dataset_pool.json` 候选池里随机选一个未用过的
3. `HfApi.dataset_info()` **预校验存在性**(HEAD 风格,跳过不存在的)
4. `datasets.load_dataset(...)` 拉取
5. 按 `text_fields` 抽纯文本 → 写 `data/extra_corpora/<ts>_<id>.txt`
6. 抽不到时**自动回退**:尝试任意 string 字段,再回退到 `conversations`/`ShareGPT`/`ChatML` 嵌套 list
7. 更新 `used_datasets.json`,把 manifest 增量 commit 回 main
8. 下次运行天然避开已用的,直到所有候选跑过一遍后再允许重复

候选池涵盖:评论、指令微调、医学推理、GitHub issues、维基嵌入语料等。
`fetch_random_data.py` 会在日志里打 `✓/✗` 标记每个候选是否真实存在。

#### 调试候选池

```bash
# 只校验,不实际拉取
python data/fetch_random_data.py --validate_only
```

输出形如:
```
[INFO] 候选池 10 个,未用过 10 个,开始预校验 ...
  ✓ seamew/ChnSentiCorp
  ✗ shibing624/chnsenti (DatasetNotFoundError: ...)
  ...
[validate_only] 7/10 个候选可用
```

根据结果,把不存在的从 `data/dataset_pool.json` 删掉或换成真实的即可。

#### 候选池字段说明

```json
{
  "id": "owner/repo",
  "config": "可选 config 名,比如 zh / cc_strict",
  "split": "train",
  "text_fields": ["question", "answer"],   // 优先拼接的字段
  "max_samples": 500,                        // 该数据集最多取多少条
  "description": "中文描述"
}
```

`text_fields` 写错也没关系,`extract_text_from_row` 会自动探测 string 字段和 conversations 嵌套。

### 5. 评估指标

`evaluate.py` 在 `data/eval.txt` 上算三件事:
- **Perplexity(PPL)**:整体 MLM 困惑度,越低越好
- **Top-1 / Top-5 准确率**:mask 后真实 token 命中率
- **Fill-Mask 定性样例**:8 个 `[MASK]` 句子的 top-5 预测

报告输出 `output/eval_report.json`,并对 base vs 微调后做对比。

### 6. 推送到 HuggingFace Hub(可选)
1. https://huggingface.co/settings/tokens 申请 **write** token
2. 仓库 `Settings → Secrets → New secret`,Name=`HF_TOKEN`
3. 手动触发时勾选 `push_to_hub=true` 并填 `hub_repo_id`

## 🔍 查看结果

每次跑完产生:

- `finetuned-model-<run_id>` —— 完整模型或 adapter + tokenizer + 训练日志 + 评估报告
- `logs-<run_id>` —— 累积的 `logs/history/`,含每天的训练日志、评估报告、摘要

下载后:
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
m = AutoModelForMaskedLM.from_pretrained("path/to/unzipped/output")
t = AutoTokenizer.from_pretrained("path/to/unzipped/output")
```

## 📦 Release 与跨日累积微调

每次成功的运行会自动创建一个 **GitHub Release**,把当前微调后的完整模型打包为 `model.zip`。
下一次运行默认从最新的 release 拉取 `model.zip` 作为新起点,实现**跨日累积微调**。

### Release 内容

每次 release 的 tag 形如 `daily-20260627T020000Z`,包含:

- **`model.zip`** — 完整 HF 模型 + tokenizer + 训练日志 + 评估报告,解压后可直接 `AutoModelForMaskedLM.from_pretrained` 加载
- **`eval_report.json`** — PPL / top-k 准确率 / fill-mask 样例
- **`training_log.json`** — 训练指标 + MoMo/LoRA 参数

Release body 包含训练摘要(模式、base、loss、耗时)、评估对比(base vs ft 的 PPL/准确率)、以及「本次基于哪个 release」。

### 跨日累积微调流程

1. **本次运行**(假设是第 1 天)
   - 从 `hfl/chinese-macbert-base` 开始微调
   - 训练 + 评估后打包 `output/` 为 `model.zip`
   - 创建 release `daily-20260627T020000Z`

2. **下次运行**(第 2 天)
   - workflow 启动后,`use_previous_model=true` 触发
   - 找到最新 release `daily-20260627T020000Z`,下载 `model.zip`
   - 解压到 `previous_model/`,作为本次 `--model_name`
   - 在第 1 天的模型基础上继续微调
   - 评估时,`--base_model` 也指向 `previous_model/`,即对比「昨天的模型 vs 今天的模型」
   - 创建 release `daily-20260628T020000Z`

3. **第 3 天、第 4 天**...依此类推,模型在每天的新数据上微调,效果逐步累积。

```
累积示意:

Day 1:  [base/macbert-base] ──微调──> Release v1
                                    ↓
Day 2:  [Release v1]        ──微调──> Release v2
                                    ↓
Day 3:  [Release v2]        ──微调──> Release v3
                                    ↓
...     (每天增量,新数据集)
```

每天的 release 都包含完整 HF 模型,任何一天都可以中断/恢复。

### 跳过累积 / 强制从 base 重来

手动触发时把 `use_previous_model` 设为 `false` 即可。模型会从 `model_name`(默认 `hfl/chinese-macbert-base`)开始。

### 手动下载并继续训练

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 1. 去 Releases 页下载最新 model.zip 并解压
model = AutoModelForMaskedLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# 2. 用你喜欢的训练代码继续微调 ...
```

## 🔧 常见问题 / Troubleshooting

### Q1: workflow 跑失败,日志说 "全部 N 次尝试都失败"

`data/fetch_random_data.py` 跑完了 `max_attempts` 个候选数据集,全部都加载/抽取失败。

**排查**:
1. 打开 Actions 日志,看每个候选的 `✓/✗` 标记
2. 去 `data/validate_dataset_pool.log`(如果存在)看每个候选的具体错误
3. 修 `data/dataset_pool.json`:把不存在的删掉,或换 `text_fields` / `config`

### Q2: 评估报 OOM / CUDA OOM

`chinese-macbert-base` ~400MB,7GB runner 单卡能跑,但全参 + 大 batch 可能 OOM。

**缓解**:
- 把 `batch_size` 从 16 调到 8 或 4
- 把 `max_len` 从 128 调到 64
- 用 `use_momo=true`(只训 ~2.4MB LoRA 专家)或 `use_lora=true`(~10MB)
- 评估时 `--batch_size 4`

### Q3: 找不到 release / `previous_model` 解压后无 `config.json`

第一次跑没有 release,正常(会从 base 模型开始)。后续跑如果 `previous_model/config.json` 缺失:

**原因**:
- 早期 release 的 `model.zip` 内容不对(比如只有 adapter 目录)
- 手动删过 release

**修复**:
- 把 `use_previous_model` 临时设为 `false` 跑一次,生成完整 release
- 或去 Releases 页确认最新 release 包含 `model.zip` 而不是只附件其他文件

### Q4: GitHub release 创建失败 "Validation Failed: tag already exists"

cron 同一分钟内触发了两次(理论上 `concurrency` 已防,极端情况下还会撞)。

**修复**:
- 删掉重复的 release 草稿
- `concurrency.cancel-in-progress: false` 改为 `true` 让新 run 取消旧 run

### Q5: 训练 loss 不下降 / NaN

- 数据太短/太脏:打开 `data/extra_corpora/`,看新数据集的样本质量
- 改小 `lr`(默认 5e-5):尝试 1e-5
- 关掉 MoMo 试试:用 `use_momo=false` 全参或 `use_lora=true`

### Q6: cron 没有触发

GitHub Actions 的 schedule 有时延(尤其在仓库一段时间没活动后),cron 会被推迟直到下一次 commit。

**检查**:Actions 页面的"Inactive schedules"会标记。

### Q7: 想停掉累积,从头来

1. 去 Releases 页 → Delete all releases
2. 把 `data/used_datasets.json` 改成 `{"used": [], "history": []}`(可选,只是清空数据集使用记录)
3. 手动触发 workflow,把 `use_previous_model` 设为 `false`

## ⚡ 性能调优建议

### 默认配置(已足够日常实验)

- `epochs=3` · `batch_size=16` · `max_len=128` · `lr=5e-5`
- MoMo: 4 experts · top-2 · r=8
- 单次完整 run(数据 + 训练 + 评估 + 打包) ~12-15 分钟(GitHub 免费 runner)

### 加速方案(按收益排序)

1. **跳 base 评估**:`skip_base_eval=true` 省 4-5 分钟
2. **缩 max_len**:从 128 调到 64,训练 token 数减半
3. **少 epochs**:从 3 调到 1,适合快速试数据效果
4. **用 MoMo 而不是全参**:训练参数量从 400M 降到 ~2.4M,迭代快 5-10x
5. **不拉新数据**:`fetch_random_data=false`,只复训已有语料

### 资源 / 成本估算

| 配置 | 训练时间 | 磁盘占用(release) | 内存峰值 |
| --- | --- | --- | --- |
| 全参 + base=macbert-base | ~10 分钟 | ~400MB(model.safetensors) | ~3GB |
| PEFT-LoRA (merge) | ~8 分钟 | ~400MB(merged) | ~2.5GB |
| MoMo (合并到 base) | ~8 分钟 | ~400MB(merged) | ~2.5GB |
| MoMo (只存 adapter) | ~8 分钟 | ~15MB(adapter) | ~2.5GB |

> GitHub 免费账户每月 2000 分钟,够跑 ~100 次完整 run。

### 训练质量调优

- **数据量不够时**:把 `max_random_samples` 从 500 调到 1000-2000
- **想学路由 / 多任务**:`momo_n_experts=8` + `momo_top_k=2`,路由更细
- **想稳定训练**:`momo_aux_alpha=0.1`(更大),防 router 坍缩
- **router 不收敛**:设环境变量 `MOMO_ROUTER_NOISE=0.1`,给路由 logits 加高斯噪声

## 🧹 清理与重置

### 清空累积的数据集(下次再从候选池抽)

```bash
# 保留 used_datasets.json 结构,只清空 used / history
echo '{"used": [], "history": []}' > data/used_datasets.json
rm -rf data/extra_corpora/*   # 可选:删掉已下载的语料,下次会重下
```

### 删掉所有 release(下次从 base 重来)

在 GitHub 网页 → Releases → Delete all releases(手动)。

或用 gh CLI:
```bash
gh release delete --yes $(gh release list --json tagName -q '.[].tagName')
```

### 清空累积的日志

```bash
rm -rf logs/history/*
```

### 完全重置项目

```bash
# 删 output / 数据缓存 / 日志
rm -rf output/ logs/ data/extra_corpora/* data/cache/
# 删所有 release(见上)
# commit:
git add -A && git commit -m "chore: full reset"
```

### 只删某个数据集的语料(但保留 used 记录)

不建议,会让 used manifest 失去对应文件;真要做就直接编辑 `data/used_datasets.json` 删掉对应 entry。

## 🔄 自动更新候选池(可选)

`data/dataset_pool.json` 写死了一组候选。**项目没有自动发现新数据集的机制**——
这是有意为之,避免在 cron 中产生不可预测的网络依赖。

如果想扩充,手动编辑 `data/dataset_pool.json` 加上新条目,然后跑:

```bash
python data/fetch_random_data.py --validate_only
```

验证新条目存在且 schema 能抽到文本。

## 🧪 本地试跑(可选,需要 Python 3.10+)

```bash
pip install -r requirements.txt

# 全参
python train.py --epochs 1 --batch_size 8
python evaluate.py

# LoRA
python train.py --use_lora --merge_lora --epochs 3
python evaluate.py

# MoMo
python train.py --use_momo --momo_n_experts 4 --momo_top_k 2 --epochs 3
python evaluate.py

# 测试随机拉数据
python data/fetch_random_data.py --max_samples 100
```

## ⚙️ 工作流关键设计

- **缓存 HuggingFace**:首次下载 ~400MB base,后续走 `actions/cache` 秒跑
- **缓存 pip**:`setup-python@v5` 自动
- **磁盘清理**:删 `dotnet` / `android` / `CodeQL`
- **时间错峰**:cron `0 2 * * *`(UTC)
- **数据自增长**:`data/extra_corpora/` + `data/used_datasets.json` 都被 commit,语料自然累积
- **跨日累积微调**:每次跑完自动创建 GitHub Release,下次默认从上一个 Release 接着练
- **Merged 完整模型优先**:`output/` 始终是合并后的完整 HF 模型,可直接加载、也可作为下一轮起点
- **训练 + 评估 + 上传 + 提交** 一站式,失败也不会丢前面的日志
- **previous_model 缓存**:跨 run 缓存上次的 `previous_model/`,配合 `restore-keys` 让 step 6.5 重新下载时不会完全冷启
- **状态可视化**:step 6.6 在训练前打 `USE_PREV` / `PREV_TAG` / `model_type`,出问题时一眼看出「这次是基于昨天的模型还是 base」

## 📝 实验性

- 训练语料基础 ~200 行,加上每日 ~300 行,大约 30 天后达到 ~9000 行
- 想要真正改善下游任务,可以把 `data/extra_corpora/` 历史数据打包再训,或换 LoRA / SFT
- 想要「版本化」累积的模型,推荐把 artifact 定期下载到本地放进 `models/v1/`、`v2/`,或推 HF Hub

## 📜 License

MIT
