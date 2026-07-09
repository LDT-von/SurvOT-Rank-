"""Slot attention components used by SurvOT-Rank methods."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from einops import pack, repeat, unpack
from einops.layers.torch import Rearrange
from torch import Tensor, einsum, nn
from torch.nn import Module, init


class MultiHeadSlotAttention(Module):
    def __init__(
        self,
        num_slots,
        dim,
        heads=4,
        dim_head=64,
        iters=3,
        eps=1e-8,
        hidden_dim=128,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

        dim_inner = dim_head * heads
        self.split_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.to_q = nn.Linear(dim, dim_inner)
        self.to_k = nn.Linear(dim, dim_inner)
        self.to_v = nn.Linear(dim, dim_inner)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")
        self.combine_heads = nn.Linear(dim_inner, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, max(dim, hidden_dim)),
            nn.ReLU(),
            nn.Linear(max(dim, hidden_dim), dim),
        )

    def forward(self, inputs, num_slots: int | None = None):
        b, _, _, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = repeat(self.slots_mu, "1 1 d -> b s d", b=b, s=n_s)
        sigma = repeat(self.slots_logsigma.exp(), "1 1 d -> b s d", b=b, s=n_s)
        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        k, v = map(self.split_heads, (k, v))

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.split_heads(self.to_q(slots))

            dots = einsum("... i d, ... j d -> ... i j", q, k) * self.scale
            attn = dots.softmax(dim=-2)
            attn = F.normalize(attn + self.eps, p=1, dim=-1)

            updates = einsum("... j d, ... i j -> ... i d", v, attn)
            updates = self.combine_heads(self.merge_heads(updates))
            updates, packed_shape = pack([updates], "* d")
            slots_prev, _ = pack([slots_prev], "* d")

            slots = self.gru(updates, slots_prev)
            (slots,) = unpack(slots, packed_shape, "* d")
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


def _log_sinkhorn_assign(
    cost: Tensor,
    max_iter: int,
    eps: float = 0.05,
    row_marginal: Optional[Tensor] = None,
    col_marginal: Optional[Tensor] = None,
) -> Tensor:
    """log-domain Sinkhorn 迭代，计算 slot↔input token 的软分配矩阵（需求5a）。

    数值稳定写法复用 `ot_event_hazard_v2/model_v2.py::log_sinkhorn_plan` 的思路：
    构造 log_mu/log_nu 边际、kernel = -cost/eps，交替用 logsumexp 更新
    log_u/log_v，最终 log_plan = kernel + log_u + log_v 再 exp() 得到分配矩阵。

    与 `log_sinkhorn_plan` 的区别：这里的 `cost` 支持任意前导批次维度
    `[..., num_slots, num_input_tokens]`（例如 Slot Attention 多头场景下的
    `[b, h, num_slots, num_tokens]`），而不仅是单个 batch 维 `[b, rows, cols]`。
    实现方式是把所有前导维度 reshape 成一个统一的 batch 维，用与
    `log_sinkhorn_plan` 完全同构的迭代公式计算，再 reshape 回原始形状。

    Args:
        cost: 代价矩阵，形状 `[..., num_slots, num_input_tokens]`。
        max_iter: 实际执行的 Sinkhorn 迭代次数上限。按需求5 AC1，取值应为
            1~1000 之间的正整数；这里选择“裁剪（clamp）”而非抛异常来处理越界
            输入，因为迭代次数是一个软性性能/精度权衡参数，裁剪不会导致
            静默的错误计算结果（分配矩阵的边际约束语义不变，只是收敛程度
            略有不同），比训练中途因为一个配置值越界而中断更安全、更不易破坏
            向后兼容性。
        eps: Sinkhorn 温度系数，默认 0.05，与 `log_sinkhorn_plan` 默认值一致。
        row_marginal: 可选的自定义行边际分布，形状需可广播到
            `[..., num_slots]`；缺省时使用均匀分布 `1/num_slots`。
        col_marginal: 可选的自定义列边际分布，形状需可广播到
            `[..., num_input_tokens]`；缺省时使用均匀分布 `1/num_input_tokens`。

    Returns:
        与 `cost` 形状相同的分配矩阵（已 exp，非 log 域）。
    """
    # 迭代次数裁剪到 [1, 1000]，见上面 max_iter 参数说明。
    max_iter = int(max(1, min(1000, max_iter)))

    orig_shape = cost.shape
    num_slots, num_tokens = orig_shape[-2], orig_shape[-1]
    device, dtype = cost.device, cost.dtype

    # 把所有前导维度合并成一个统一的 batch 维，复用与 log_sinkhorn_plan 同构的
    # 二维（按 batch）迭代公式。
    cost_flat = cost.reshape(-1, num_slots, num_tokens)
    bsz = cost_flat.shape[0]

    if row_marginal is None:
        log_mu = torch.full(
            (bsz, num_slots), 1.0 / num_slots, device=device, dtype=dtype
        ).log()
    else:
        # 用标准广播规则（从右侧对齐）把自定义行边际扩展到 [..., num_slots]，
        # 再展平成统一 batch 维，与默认路径保持同样的形状契约。
        row_marginal_b = torch.broadcast_to(row_marginal, orig_shape[:-1])
        log_mu = row_marginal_b.reshape(bsz, num_slots).to(dtype=dtype).log()

    if col_marginal is None:
        log_nu = torch.full(
            (bsz, num_tokens), 1.0 / num_tokens, device=device, dtype=dtype
        ).log()
    else:
        col_marginal_shape = (*orig_shape[:-2], num_tokens)
        col_marginal_b = torch.broadcast_to(col_marginal, col_marginal_shape)
        log_nu = col_marginal_b.reshape(bsz, num_tokens).to(dtype=dtype).log()

    kernel = -cost_flat / eps
    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros_like(log_nu)

    for _ in range(max_iter):
        log_u = log_mu - torch.logsumexp(kernel + log_v.unsqueeze(1), dim=2)
        log_v = log_nu - torch.logsumexp(kernel + log_u.unsqueeze(2), dim=1)

    log_plan = kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)
    plan = log_plan.exp()
    return plan.reshape(orig_shape)


class MultiHeadSlotAttentionV2(Module):
    """支持 identity/state 解耦（以及后续任务将接入的 Sinkhorn 路由/跨模态条件化/自适应
    迭代次数）的 Slot Attention 变体。

    本类只新增，不修改现有 `MultiHeadSlotAttention` 的任何行为。

    Args:
        num_slots, dim, heads, dim_head, iters, eps, hidden_dim: 与
            `MultiHeadSlotAttention` 完全相同的含义。
        use_disentangled_slots: 是否启用 identity/state 解耦（需求4）。默认 False 时，
            前向传播逐位复用 `MultiHeadSlotAttention.forward` 的现有迭代逻辑
            （softmax 竞争 + 固定 iters + GRU 更新），与旧实现数值一致。
        router: 路由机制，"softmax"（默认，当前行为）| "sinkhorn"（需求5a，将在后续任务
            中接入，本任务仅预留该配置字段，不改变前向逻辑）。
        sinkhorn_max_iters: Sinkhorn 路由的最大迭代次数上限（需求5 AC1，后续任务接入）。
        cross_modal_conditioning: 是否启用跨模态条件化更新（需求5b，后续任务接入）。
        adaptive_iters: 是否启用自适应迭代次数（需求5c，后续任务接入）。
        convergence_threshold: 自适应迭代的收敛阈值（后续任务接入）。
        max_iters_cap: 自适应迭代模式下的迭代次数上限（后续任务接入）。
    """

    def __init__(
        self,
        num_slots,
        dim,
        heads=4,
        dim_head=64,
        iters=3,
        eps=1e-8,
        hidden_dim=128,
        use_disentangled_slots: bool = False,
        router: str = "softmax",
        sinkhorn_max_iters: int = 20,
        cross_modal_conditioning: bool = False,
        adaptive_iters: bool = False,
        convergence_threshold: float = 0.0,
        max_iters_cap: int = 10,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        # 以下字段是为后续任务（2.1 Sinkhorn 路由、3.1 跨模态条件化、4.1 自适应迭代次数）
        # 预留的配置项。本任务（1.1）只消费 use_disentangled_slots；其余字段的默认值
        # 保证在未被后续任务接入前，前向逻辑与当前 MultiHeadSlotAttention 完全一致。
        self.use_disentangled_slots = use_disentangled_slots
        self.router = router
        self.sinkhorn_max_iters = sinkhorn_max_iters
        self.cross_modal_conditioning = cross_modal_conditioning
        self.adaptive_iters = adaptive_iters
        self.convergence_threshold = convergence_threshold
        self.max_iters_cap = max_iters_cap

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

        dim_inner = dim_head * heads
        self.split_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.to_q = nn.Linear(dim, dim_inner)
        self.to_k = nn.Linear(dim, dim_inner)
        self.to_v = nn.Linear(dim, dim_inner)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")
        self.combine_heads = nn.Linear(dim_inner, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, max(dim, hidden_dim)),
            nn.ReLU(),
            nn.Linear(max(dim, hidden_dim), dim),
        )

        # Identity/State 解耦（需求4）：
        # - slot_identity: [num_slots, dim]，batch 间共享，跨迭代步骤不变（Slot_Identity）。
        # - slot_state（前向传播中的迭代变量，即原 `slots`）：[batch, num_slots, dim]，
        #   由输入内容与 GRU 迭代更新决定（Slot_State）。
        self.slot_identity = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.identity_proj = nn.Linear(dim * 2, dim)

        # 跨模态条件化更新（需求5b）：始终实例化该层（即便
        # cross_modal_conditioning=False 时也不使用），保证 state_dict 的键集合
        # 与开关状态无关，简化对比/加载逻辑。
        self.cross_modal_proj = nn.Linear(dim, dim)

    def forward(
        self,
        inputs,
        num_slots: int | None = None,
        cross_modal_state: Optional[Tensor] = None,
    ):
        b, _, _, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = repeat(self.slots_mu, "1 1 d -> b s d", b=b, s=n_s)
        sigma = repeat(self.slots_logsigma.exp(), "1 1 d -> b s d", b=b, s=n_s)
        slot_state = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        k, v = map(self.split_heads, (k, v))

        for t in range(self.max_iters_cap):
            slot_state_prev = slot_state
            slot_state = self.norm_slots(slot_state)
            q = self.split_heads(self.to_q(slot_state))

            dots = einsum("... i d, ... j d -> ... i j", q, k) * self.scale

            if self.router == "sinkhorn":
                # 需求5a：用 log-domain Sinkhorn 分配矩阵替代 softmax 竞争。
                # dots 形状为 [b, h, num_slots, num_tokens]；cost 取 -dots，使得
                # 相似度越高（dots 越大）代价越低，Sinkhorn 分配概率越大，
                # 与 softmax(dim=-2) 让 token 在 slot 间竞争的语义保持一致方向。
                attn = _log_sinkhorn_assign(
                    cost=-dots, max_iter=self.sinkhorn_max_iters, eps=0.05
                )
                # Sinkhorn 分配矩阵已经满足行/列边际约束（行和≈1/num_slots，
                # 列和≈1/num_tokens），因此不再执行下面 softmax 路径的
                # `F.normalize(attn + eps, p=1, dim=-1)` 按行重归一化——
                # 那一步是为了让 softmax(dim=-2) 之后 token 维度上的和为 1
                # （因为 softmax 是在 slot 维度上做的，需要额外沿 token 维度归一化
                # 才能得到"每个 slot 的注意力权重和为1"）。Sinkhorn 分配矩阵本身
                # 已经是一个满足两侧边际的联合分布，再做一次 L1 归一化会破坏其中
                # 一侧的边际约束（这也是下面 (c) 项验证 1e-3 容差的前提）。
            else:
                attn = dots.softmax(dim=-2)
                attn = F.normalize(attn + self.eps, p=1, dim=-1)

            updates = einsum("... j d, ... i j -> ... i d", v, attn)
            updates = self.combine_heads(self.merge_heads(updates))

            if self.cross_modal_conditioning and cross_modal_state is not None:
                # 需求5b：跨模态条件化更新。cross_modal_state 形状为
                # [batch, num_slots, dim]，与 updates 此时的形状一致，
                # 直接加进 GRU 输入之前的更新量，保证换一个不同的
                # cross_modal_state 会让输出数值变化（需求5 AC9）。
                updates = updates + self.cross_modal_proj(cross_modal_state)

            updates, packed_shape = pack([updates], "* d")
            slot_state_prev_packed, _ = pack([slot_state_prev], "* d")

            slot_state = self.gru(updates, slot_state_prev_packed)
            (slot_state,) = unpack(slot_state, packed_shape, "* d")
            slot_state = slot_state + self.mlp(self.norm_pre_ff(slot_state))

            if self.adaptive_iters:
                criterion = (slot_state - slot_state_prev).norm(p=2)
                if t >= 1 and criterion < self.convergence_threshold:
                    break
            elif t + 1 >= self.iters:
                break

        if self.use_disentangled_slots:
            identity = repeat(self.slot_identity, "s d -> b s d", b=b)
            output = self.identity_proj(torch.cat([slot_state, identity], dim=-1))
        else:
            output = slot_state

        return output


def _log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -_log(-_log(noise))


def relaxed_topk(logits, k, temperature=1.0):
    scores = logits
    soft_k_hot = torch.zeros_like(logits)
    for _ in range(k):
        probs = F.softmax(scores / temperature, dim=-1)
        soft_k_hot = soft_k_hot + probs
        scores = scores + torch.log((1.0 - probs).clamp(min=1e-20))
    return soft_k_hot


def gumbel_topk_st(logits, k=1, temperature=1.0):
    noised_logits = logits + gumbel_noise(logits)
    topk_indices = noised_logits.topk(k=k, dim=-1).indices
    hard_k_hot = torch.zeros_like(logits)
    hard_k_hot.scatter_(1, topk_indices, 1.0)
    soft_k_hot = relaxed_topk(noised_logits, k=k, temperature=temperature)
    return hard_k_hot + soft_k_hot - soft_k_hot.detach(), topk_indices


def parallel_topk_st(logits, k=1, temperature=1.0):
    noised_logits = logits + gumbel_noise(logits)
    topk_indices = noised_logits.topk(k=k, dim=-1).indices
    hard_k_hot = torch.zeros_like(logits)
    hard_k_hot.scatter_(1, topk_indices, 1.0)
    soft_k_hot = k * F.softmax(noised_logits / temperature, dim=-1)
    return hard_k_hot + soft_k_hot - soft_k_hot.detach(), topk_indices



def build_slot_attention(dim, num_slots, heads, iters, config):
    """工厂函数：根据 config 中的新增可选字段决定返回原始 `MultiHeadSlotAttention`
    还是配置好的 `MultiHeadSlotAttentionV2`（需求5 AC8，需求7 AC1）。

    Args:
        dim, num_slots, heads, iters: 与两个类构造函数中的同名参数一致。
        config: 一个具有（可能缺失）`otehv2v2_slot_router`、`otehv2v2_slot_disentangled`、
            `otehv2v2_slot_cross_modal_cond`、`otehv2v2_slot_adaptive_iters`、
            `otehv2v2_sinkhorn_max_iters`、`otehv2v2_convergence_threshold` 等属性的对象
            （通常是 args Namespace），用 `getattr(config, name, default)` 读取。

    Returns:
        当所有新增字段均为默认值（softmax 路由、无解耦、无跨模态条件化、无自适应迭代）时，
        返回一个 `MultiHeadSlotAttention` 实例（与当前 V45 baseline 完全一致）；否则返回
        一个按 config 配置好的 `MultiHeadSlotAttentionV2` 实例。
    """
    router = getattr(config, "otehv2v2_slot_router", "softmax")
    disentangled = getattr(config, "otehv2v2_slot_disentangled", False)
    cross_modal_cond = getattr(config, "otehv2v2_slot_cross_modal_cond", False)
    adaptive_iters = getattr(config, "otehv2v2_slot_adaptive_iters", False)

    if router == "softmax" and not disentangled and not cross_modal_cond and not adaptive_iters:
        return MultiHeadSlotAttention(dim=dim, num_slots=num_slots, heads=heads, iters=iters)

    sinkhorn_max_iters = getattr(config, "otehv2v2_sinkhorn_max_iters", 20)
    convergence_threshold = getattr(config, "otehv2v2_convergence_threshold", 0.0)
    max_iters_cap = max(iters, 10)

    return MultiHeadSlotAttentionV2(
        dim=dim,
        num_slots=num_slots,
        heads=heads,
        iters=iters,
        use_disentangled_slots=disentangled,
        router=router,
        sinkhorn_max_iters=sinkhorn_max_iters,
        cross_modal_conditioning=cross_modal_cond,
        adaptive_iters=adaptive_iters,
        convergence_threshold=convergence_threshold,
        max_iters_cap=max_iters_cap,
    )
