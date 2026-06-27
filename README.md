# GR — 每日中文 BERT 自动微调 + 评估

> 利用 **GitHub Actions** 每日自动跑一次中文 BERT 的轻量微调与评估。
> 这是一个实验性项目,目的是**探究与利用 GitHub** 的免费计算能力。

## ✨ 项目目标

- 每天 0 点(北京时间)自动在 GitHub 云端跑一次模型微调
- 训练后**自动评估** base 与微调后模型,产出 PPL / top-k 准确率
- 支持**全参微调**与**LoRA 微调**两种模式
- 不需要本地 Python 环境
- 模型上传为 artifact,可下载、对比、观察逐步演化

## 🤖 选型

| 项目 | 选择 | 理由 |
| --- | --- | --- |
| Base 模型 | **`hfl/chinese-macbert-base`** | 纯 PyTorch,中文原生,~400MB,MIT 协议,中文效果优于 `bert-base-chinese` |
| 备选 | `bert-base-chinese` / `hfl/chinese-roberta-wwm-ext` | 同样可用,直接换 `--model_name` |
| 任务 | **Masked Language Modeling** | 适合 BERT 系列,可增量训练,无需标注数据 |
| 数据 | `data/corpus.txt`(训练) + `data/eval.txt`(评估) | 古诗、现代文、谚语、技术文本混排 |
| Runner | `ubuntu-latest`(免费,2 核 7G) | 够用,加缓存后单次训练 < 10 分钟 |

## 📂 仓库结构

```
GR/
├── .github/workflows/daily-finetune.yml   # 每日 cron 工作流
├── data/
│   ├── corpus.txt                         # 训练语料
│   └── eval.txt                           # 评估语料
├── train.py                               # 训练(支持 LoRA)
├── evaluate.py                            # 评估(PPL + top-k + fill-mask)
├── requirements.txt                       # Python 依赖
├── README.md
└── .gitignore
```

## 🚀 使用方式

### 1. 直接使用
- Fork / push 到你的 GitHub 仓库
- 启用 GitHub Actions
- 每天 UTC 02:00 自动跑一次(北京时间 10:00)
- 也可在 Actions 页面手动触发 `Daily Chinese BERT Fine-tune + Evaluate`

### 2. 手动触发参数

| 输入 | 默认 | 说明 |
| --- | --- | --- |
| `model_name` | `hfl/chinese-macbert-base` | 任何纯 PyTorch 中文模型 |
| `epochs` | `3` | 训练轮数 |
| `batch_size` | `16` | 每设备 batch size |
| `use_lora` | `false` | 启用 LoRA 微调(PEFT) |
| `lora_r` | `8` | LoRA 秩 |
| `lora_alpha` | `16` | LoRA alpha |
| `lora_dropout` | `0.1` | LoRA dropout |
| `merge_lora` | `true` | 训练后把 LoRA 合并回 base(evaluate.py 不需感知 LoRA) |
| `skip_base_eval` | `false` | 跳过 base 模型评估(只评微调后,加速) |
| `push_to_hub` | `false` | 是否推送到 HuggingFace Hub |
| `hub_repo_id` | 空 | HF Hub 仓库名,如 `Annieif/gr-macbert` |

### 3. 训练模式:全参 vs LoRA

**全参微调**(默认,~400MB checkpoint):
- 训练所有参数,效果上限高
- 适合:有充足 GPU / 想学完整范式

**LoRA 微调**(`--use_lora`,~10MB adapter):
- 只训练低秩适配器 + MLM 头
- 适合:**频繁日训**、磁盘/内存紧张、想保留多版本 adapter
- 默认 `--merge_lora`,合并后保存为完整模型,使用上和全参完全一致
- 想保留纯 adapter(几十 MB),加 `--no-merge`(`workflow_dispatch` 里把 `merge_lora` 改 `false`)

### 4. 评估指标

`evaluate.py` 在 `data/eval.txt` 上算三件事:

1. **Perplexity(PPL)**:整体语言建模困惑度,越低越好
2. **Top-k Masked Token Accuracy**:MLM 任务上 top-1 / top-5 命中真实 token 的比例
3. **Fill-Mask 定性样例**:8 个 `[MASK]` 句子,展示前 5 个预测,直观对比 base vs 微调

报告输出到 `output/eval_report.json`,关键字段:

```json
{
  "base":      {"perplexity": 87.4, "top1_acc": 0.41, "top5_acc": 0.69, ...},
  "finetuned": {"perplexity": 73.2, "top1_acc": 0.46, "top5_acc": 0.74, ...},
  "comparison": {
    "perplexity": {"base": 87.4, "ft": 73.2, "delta": -14.2, "better": "ft"}
  }
}
```

### 5. 推送到 HuggingFace Hub(可选)
1. 在 https://huggingface.co/settings/tokens 申请一个 **write** token
2. 仓库 `Settings → Secrets and variables → Actions → New repository secret`
3. Name: `HF_TOKEN`,Value: 你的 token
4. 手动触发 workflow 时勾选 `push_to_hub=true` 并填 `hub_repo_id`

### 6. 替换或扩充训练/评估数据
- 训练数据:`data/corpus.txt`,一行一句 UTF-8 中文
- 评估数据:`data/eval.txt`,一行一句 UTF-8 中文,**应与训练集不重叠**
- 越大越多样,微调效果越明显

## 🔍 查看结果

每次 workflow 跑完会产生两个 artifact:

- `finetuned-model-<run_id>` —— 完整模型(全参或 LoRA-merged)+ tokenizer + 训练日志 + 评估报告
- `logs-<run_id>` —— 累积的 `logs/history/`,含每天的训练日志、评估报告、摘要

下载 zip 后可直接用 `transformers` 加载:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
m = AutoModelForMaskedLM.from_pretrained("path/to/unzipped/output")
t = AutoTokenizer.from_pretrained("path/to/unzipped/output")
```

## 🧪 本地试跑(可选,需要 Python 3.10+)

```bash
pip install -r requirements.txt

# 全参微调 + 评估
python train.py --epochs 1 --batch_size 8
python evaluate.py

# LoRA 微调 + 评估
python train.py --use_lora --merge_lora --epochs 3
python evaluate.py
```

## ⚙️ 工作流关键设计

- **缓存 HuggingFace**: 第一次跑会下载 ~400MB base 模型,后续用 `actions/cache` 命中,几乎秒跑
- **缓存 pip**: 由 `setup-python@v5` 的 `cache: 'pip'` 自动处理
- **磁盘清理**: 删掉 `dotnet` / `android` / `CodeQL`,从 30G 释放到 ~25G
- **时间错峰**: cron 设为 `0 2 * * *`(UTC),避开 GitHub 资源高峰
- **手动触发**: 支持 `workflow_dispatch` 调试,可改模型/LoRA/上传 HF 等
- **日志持久化**: 训练日志 + 评估报告 + 摘要 通过 `git commit` 推回 main,留痕每一天的训练/评估曲线

## 📝 实验性

- 训练语料只有 ~200 行,效果有限 —— 这是**有意的**,目的是看「每日小步微调 + 累积」能不能跑通
- 想要真正改善下游任务,请扩充 `data/corpus.txt` / `data/eval.txt`,或换成 LoRA / SFT 任务
- 想要「版本化」累积的模型,推荐把 artifact 定期下载到本地,放进 `models/v1/`、`v2/` 之类的目录,或推到 HF Hub

## 📜 License

MIT
