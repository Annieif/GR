# GR — 每日中文 BERT 自动微调 + 评估(MoE+Multi-LoRA)

> 利用 **GitHub Actions** 每日自动跑一次中文 BERT 的轻量微调与评估。
> 每次运行会从候选池**随机抽取一个未用过的中文数据集**作为新训练素材,
> 配合 **MoE + Multi-LoRA(MoMo)**,让模型在「数据 + 架构」两个维度随时间演化。
> 这是一个实验性项目,目的是**探究与利用 GitHub** 的免费计算能力。

## ✨ 项目目标

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
3. `datasets.load_dataset(...)` 拉取
4. 抽若干条纯文本 → 写到 `data/extra_corpora/<ts>_<id>.txt`
5. 更新 `used_datasets.json`,把 manifest 增量 commit 回 main
6. 下次运行天然避开已用的,直到所有候选跑过一遍后再允许重复

候选池涵盖:古诗、文学评论、GitHub issues、指令微调、医学/中医/心理学问答、
分类语料、论文摘要等。`fetch_random_data.py` 内置重试,最多试 4 个候选。

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
- **训练 + 评估 + 上传 + 提交** 一站式,失败也不会丢前面的日志

## 📝 实验性

- 训练语料基础 ~200 行,加上每日 ~300 行,大约 30 天后达到 ~9000 行
- 想要真正改善下游任务,可以把 `data/extra_corpora/` 历史数据打包再训,或换 LoRA / SFT
- 想要「版本化」累积的模型,推荐把 artifact 定期下载到本地放进 `models/v1/`、`v2/`,或推 HF Hub

## 📜 License

MIT
