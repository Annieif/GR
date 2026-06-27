"""
MoE + Multi-LoRA (MoLoRA) 注入模块
===================================

把 BERT(或其他 HF 模型)中指定的 `nn.Linear` 替换为:
  frozen base  +  N 个 LoRA experts  +  一个 router(gating network)

每个 token 由 router 选出 top-k 个 expert,LoRA 输出按权重相加。
训练时同步加一个 load-balancing 辅助 loss,避免 router 坍缩到 1-2 个 expert。

典型用法:
    from momo_lora import inject_momo_lora, add_momo_aux_loss_hook, get_momo_param_count

    model = AutoModelForMaskedLM.from_pretrained(...)
    n = inject_momo_lora(model,
                         target_module_names=["query", "value"],
                         n_experts=4, top_k=2,
                         lora_r=8, lora_alpha=16)
    print(f"替换了 {n} 个 Linear -> MoLoRALinear")
    print(get_momo_param_count(model))
    add_momo_aux_loss_hook(model)
    # 之后直接用 transformers.Trainer 训练
"""
from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoLoRALinear(nn.Module):
    """
    替换 nn.Linear:frozen base + N 个 LoRA expert + top-k router。
    """

    def __init__(self,
                 base: nn.Linear,
                 n_experts: int = 4,
                 top_k: int = 2,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.0,
                 aux_loss_alpha: float = 0.01):
        super().__init__()
        # base 冻结
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.n_experts = int(n_experts)
        self.top_k = min(int(top_k), self.n_experts)
        self.lora_r = int(lora_r)
        self.scaling = lora_alpha / self.lora_r
        self.aux_loss_alpha = float(aux_loss_alpha)

        # N 个 LoRA expert,参数化形式 (n_experts, r, in) 和 (n_experts, out, r)
        self.lora_A = nn.Parameter(torch.zeros(self.n_experts, self.lora_r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.n_experts, self.out_features, self.lora_r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(lora_dropout)

        # Router:hidden -> n_experts
        self.router = nn.Linear(self.in_features, self.n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        # 上次前向的辅助 loss(被 add_momo_aux_loss_hook 读取)
        self._last_aux_loss: torch.Tensor | None = None

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base 输出(冻结)
        base_out = self.base(x)

        # 路由
        router_logits = self.router(x)                            # [..., n_experts]
        top_logits, top_idx = router_logits.topk(self.top_k, dim=-1)  # [..., k]
        top_probs = F.softmax(top_logits, dim=-1)                 # [..., k]

        # 计算所有 expert 的 LoRA 增量
        #   lora_A: (n, r, in), lora_B: (n, out, r)
        #   hidden = x @ A.T  -> (..., n, r)
        #   out_per_expert = hidden @ B.T -> (..., n, out)
        x_drop = self.lora_dropout(x)
        hidden = torch.einsum('...i,nri->...nr', x_drop, self.lora_A)   # (..., n, r)
        out_per_expert = torch.einsum('...nr,noi->...no', hidden, self.lora_B)  # (..., n, out)
        out_per_expert = out_per_expert * self.scaling

        # 路由权重:对每个 expert,累加它在 top-k 中被选中的概率
        #   top_idx: (..., k),  top_probs: (..., k)
        #   one_hot: (..., k, n) -> sum over k weighted by top_probs -> (..., n)
        one_hot = F.one_hot(top_idx, num_classes=self.n_experts).to(top_probs.dtype)
        expert_w = (one_hot * top_probs.unsqueeze(-1)).sum(dim=-2)         # (..., n)
        expert_out = (out_per_expert * expert_w.unsqueeze(-1)).sum(dim=-2)  # (..., out)

        out = base_out + expert_out

        # 辅助 loss(仅在训练时)
        if self.training and self.aux_loss_alpha > 0:
            self._last_aux_loss = self._compute_aux_loss(router_logits)
        else:
            self._last_aux_loss = None

        return out

    # ------------------------------------------------------------------ aux
    def _compute_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Switch-Transformer 风格的负载均衡 loss:
            L_aux = alpha * N * sum_i ( f_i * P_i )
        其中 f_i = token 被路由到 expert i 的比例, P_i = router 给 expert i 的平均概率。
        """
        with torch.no_grad():
            top1_idx = router_logits.argmax(dim=-1)             # [...]
            flat_idx = top1_idx.reshape(-1)
            n_tokens = flat_idx.numel()
            f = torch.zeros(self.n_experts, device=router_logits.device, dtype=torch.float32)
            for i in range(self.n_experts):
                f[i] = (flat_idx == i).float().sum() / max(n_tokens, 1)
        P = F.softmax(router_logits.float(), dim=-1)
        # 对所有非 expert 维求平均
        P = P.mean(dim=tuple(range(P.dim() - 1)))               # [n_experts]
        loss = self.aux_loss_alpha * self.n_experts * (f * P).sum()
        return loss.to(router_logits.dtype)


# ---------------------------------------------------------------------- inject
def inject_momo_lora(model: nn.Module,
                     target_module_names: Iterable[str],
                     n_experts: int = 4,
                     top_k: int = 2,
                     lora_r: int = 8,
                     lora_alpha: int = 16,
                     lora_dropout: float = 0.0,
                     aux_loss_alpha: float = 0.01,
                     skip_if_existing: bool = True) -> int:
    """
    遍历 model,把名字匹配 target_module_names 的 nn.Linear 替换为 MoLoRALinear。
    默认只处理叶子 Linear(避免把整个 attention 干碎)。
    """
    targets = set(target_module_names)
    n_replaced = 0
    # 用 list() 拷贝避免迭代时修改
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in targets:
                continue
            if skip_if_existing and isinstance(child, MoLoRALinear):
                continue
            new = MoLoRALinear(
                child,
                n_experts=n_experts, top_k=top_k,
                lora_r=lora_r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                aux_loss_alpha=aux_loss_alpha,
            )
            setattr(parent, child_name, new)
            n_replaced += 1
    return n_replaced


# ---------------------------------------------------------------------- hook
def add_momo_aux_loss_hook(model: nn.Module) -> None:
    """
    包装 model.forward,把模型里所有 MoLoRALinear 的 _last_aux_loss
    累加到返回对象的 loss 上。transformers.Trainer 期望这种结构。
    """
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        # 既可能是 ModelOutput(含 .loss),也可能只有 logits
        loss = getattr(out, "loss", None)
        if loss is None:
            return out
        aux_total = None
        n_aux = 0
        for m in model.modules():
            if isinstance(m, MoLoRALinear) and m._last_aux_loss is not None:
                if aux_total is None:
                    aux_total = m._last_aux_loss
                else:
                    aux_total = aux_total + m._last_aux_loss
                n_aux += 1
        if aux_total is not None and n_aux > 0:
            out.loss = loss + aux_total
        return out

    model.forward = wrapped_forward


# ---------------------------------------------------------------------- stats
def get_momo_param_count(model: nn.Module) -> dict:
    """统计 total / trainable / momo 参数。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_momo = 0
    momo_params = 0
    for m in model.modules():
        if isinstance(m, MoLoRALinear):
            n_momo += 1
            momo_params += sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
        "n_momo_layers": n_momo,
        "momo_params": momo_params,
    }


# ---------------------------------------------------------------------- save
def save_momo_checkpoint(model: nn.Module, tokenizer, output_dir: str,
                       base_model_name: str | None = None) -> None:
    """
    保存:
      output_dir/
        ├── adapter/                 # 纯 MoLoRA 参数(router + lora_A/B)
        ├── tokenizer files
        └── (可选)model.safetensors  完整 base 权重(若 freeze_save_base=True)
    evaluate.py 可识别这种布局并自动 load。
    """
    import os
    import json
    from pathlib import Path
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 MoLoRA 参数
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, MoLoRALinear):
            state[f"{name}.lora_A"] = m.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = m.lora_B.detach().cpu()
            state[f"{name}.router.weight"] = m.router.weight.detach().cpu()

    save_file(state, str(adapter_dir / "adapter_model.safetensors"))

    # 配置
    cfg = {
        "model_type": "momo_lora",
        "base_model_name_or_path": base_model_name,
        "n_experts": getattr(model, "_momo_n_experts", None),
        "top_k": getattr(model, "_momo_top_k", None),
        "lora_r": getattr(model, "_momo_lora_r", None),
        "lora_alpha": getattr(model, "_momo_lora_alpha", None),
        "lora_dropout": getattr(model, "_momo_lora_dropout", None),
        "aux_loss_alpha": getattr(model, "_momo_aux_loss_alpha", None),
        "target_module_names": getattr(model, "_momo_targets", None),
    }
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # tokenizer
    tokenizer.save_pretrained(str(output_dir))

    print(f"[INFO] MoLoRA adapter saved to {adapter_dir} ({len(state)} tensors)")


def load_momo_checkpoint(model: nn.Module, adapter_dir: str,
                         base_model_name: str | None = None) -> int:
    """
    把保存的 MoLoRA 参数灌回已经注入好 MoLoRALinear 结构的 model。
    返回成功加载的 tensor 数。
    """
    import os
    from pathlib import Path
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    sf_path = adapter_dir / "adapter_model.safetensors"
    if not sf_path.exists():
        raise FileNotFoundError(f"未找到 {sf_path}")

    state = load_file(str(sf_path))
    own = dict(model.named_parameters())
    n_loaded = 0
    for k, v in state.items():
        if k in own:
            own[k].data.copy_(v.to(own[k].device))
            n_loaded += 1
    print(f"[INFO] MoLoRA loaded {n_loaded}/{len(state)} tensors from {sf_path}")
    return n_loaded
