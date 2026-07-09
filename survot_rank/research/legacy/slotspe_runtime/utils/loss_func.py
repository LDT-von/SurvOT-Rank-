
import torch.nn as nn
import torch
import torch.nn.functional as F

class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h) #hazard function
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)

    # print("S_padded.shape", S_padded.shape, S_padded)


    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case 
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


def loss_reg_l1(coef):
    print('[setup] L1 loss with coef={}'.format(coef))
    coef = .0 if coef is None else coef
    def func(model_params):
        if coef <= 1e-8:
            return 0.0
        else:
            return coef * sum([torch.abs(W).sum() for W in model_params])
    return func

##################################################
# General Loss for Survival Analysis Models,
# including continuous output and discrete output.
##################################################

def recon_loss(pred_t, t, e, alpha=0.0, gamma=1.0, norm='l1', cur_alpha=None):
    """Continuous Survival Model

    Reconstruction loss for pred_t and labels.
    recon_loss = l2 + l3
    if e = 0, l2 = max(0, t - pred_t)
    if e = 1, l3 = |t - pred_t|
    """
    pred_t = pred_t.squeeze()
    t = t.squeeze()
    e = e.squeeze()
    loss_obs = e * torch.abs(pred_t - t)
    loss_cen = (1 - e) * F.relu(gamma - (pred_t - t))
    if norm == 'l2':
        loss_obs = loss_obs * loss_obs
        loss_cen = loss_cen * loss_cen
    loss_recon = loss_obs + loss_cen
    _alpha = alpha if cur_alpha is None else cur_alpha
    loss = (1.0 - _alpha) * loss_recon + _alpha * loss_obs
    loss = loss.mean()
    return loss

class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()
        self.gamma = 1.0
        self.norm = 'l1'
        self.add_weight = False

    def forward(self, pred_t, t, e):
        return rank_loss(pred_t, t, e, self.gamma, self.norm, self.add_weight)

def rank_loss(pred_t, t, e, gamma=1, norm='l1', add_weight=False):
    """Continuous Survival Model

    Ranking loss for preditions and observations.
    for pairs (i, j) conditioned on e_i = 1 & t_i < t_j:
        diff_ij = (-pred_t_i) - (-pred_t_j)
        rank_loss = ||max(0, gamma - diff_ij)||_norm
                  = ||max(0, gamma + pred_t_i - pred_t_j)||_norm
    """
    # pred_t = pred_t.squeeze()
    hazards = torch.sigmoid(pred_t)
    survival = torch.cumprod(1 - hazards, dim=1)
    pred_t = -torch.sum(survival, dim=1)
    t = t.squeeze()
    e = e.squeeze()
    pair_mask = (t.view(-1, 1) < t.view(1, -1)) * (e.view(-1, 1) == 1)
    if not torch.any(pair_mask):
        return torch.Tensor([0.0]).to(pred_t.device)
    pair_diff = pred_t.view(-1, 1) - pred_t.view(1, -1) # the lower, the best
    pair_loss = F.relu(gamma + pair_diff)
    pair_mask = pair_mask.float()
    if add_weight:
        # masked_log_softmax
        x = pair_diff
        maxx = (x * pair_mask + (1 - 1 / (pair_mask + 1e-5))).max()
        log_ex = x - maxx
        log_softmax = log_ex - (torch.exp(log_ex * pair_mask) * pair_mask).sum().log()
        normed_weight = (log_softmax * pair_mask).exp() * pair_mask
    else:
        weight = pair_mask
        normed_weight = weight / weight.sum()

    if norm == 'l2':
        pair_loss = pair_loss * pair_loss
    elif norm == 'l1':
        pass
    else:
        raise NotImplementedError('Arg. `norm` expected l1/l2, but got {}'.format(norm))

    rank_loss = (pair_loss * normed_weight).sum()
    return rank_loss

def MSE_loss(pred_t, t, e, include_censored=False):
    """Continuous Survival Model.

    MSE loss for pred_t and labels, used for reproducing ESAT (shen et al., ESAT, AAAI, 2022).
    Please refer to its official repo: https://github.com/notbadforme/ESAT/blob/main/esat/trainforesat.py#L111
    """
    pred_t = pred_t.squeeze()
    t = t.squeeze()
    e = e.squeeze()
    loss = e * (pred_t - t) * (pred_t - t)
    if include_censored:
        loss += (1 - e) * (pred_t - t) * (pred_t - t)
    loss = loss.mean()
    return loss

class SurvMLE(nn.Module):
    """A maximum likelihood estimation function in Survival Analysis.
    As suggested in '10.1109/TPAMI.2020.2979450',
        [*] L = (1 - alpha) * loss_l + alpha * loss_z.
    where loss_l is the negative log-likelihood loss, loss_z is an upweighted term for instances
    D_uncensored. In discrete model, T = 0 if t in [0, a_1), T = 1 if t in [a_1, a_2) ...
    The larger the alpha, the bigger the importance of event_loss.
    If alpha = 0, event loss and censored loss are viewed equally.
    This implementation is based on https://github.com/mahmoodlab/MCAT/blob/master/utils/utils.py
    """
    def __init__(self, alpha=0.0, eps=1e-7):
        super(SurvMLE, self).__init__()
        self.alpha = alpha
        self.eps = eps
        print('[setup] loss: a MLE loss in discrete SA models with alpha = %.2f' % self.alpha)

    def forward(self, hazards_hat, t, e, cur_alpha=None):
        """
        y: torch.FloatTensor() with shape of [B, 2] for a discrete model.
        t: torch.LongTensor() with shape of [B, ] or [B, 1]. It's a discrete time label.
        e: torch.FloatTensor() with shape of [B, ] or [B, 1].
            e = 1 for uncensored samples (with event),
            e = 0 for censored samples (without event).
        hazards_hat: torch.FloatTensor() with shape of [B, MAX_T]
        """
        batch_size = len(t)
        t = t.view(batch_size, 1).long() # ground truth bin, 0 [0,a_1), 1 [a_1,a_2),...,k-1 [a_k-1,inf)
        c = 1 - e.view(batch_size, 1).float() # convert it to censorship status, 0 or 1
        S = torch.cumprod(1 - hazards_hat, dim=1) # surival is cumulative product of 1 - hazards
        S_padded = torch.cat([torch.ones_like(c), S], 1) # s[0] = 1.0 to avoid for t = 0
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, t).clamp(min=self.eps)) + torch.log(torch.gather(hazards_hat, 1, t).clamp(min=self.eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, t+1).clamp(min=self.eps))
        neg_l = censored_loss + uncensored_loss
        alpha = self.alpha if cur_alpha is None else cur_alpha
        loss = (1.0 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss

class SurvPLE(nn.Module):
    """
    Partial Likelihood Estimation loss for Cox model.
    Now supports: E = 1 means censored; E = 0 means event observed.
    """

    def __init__(self):
        super(SurvPLE, self).__init__()
        self.CONSTANT = torch.tensor(10.0)
        print('[setup] loss: PLE loss with E=1 as censored')

    def forward(self, y_hat, T, E):
        y_hat = torch.sigmoid(y_hat)
        device = y_hat.device

        # prevent overflow in exp
        cont = self.CONSTANT.to(device)
        y_hat = torch.where(y_hat > cont, cont, y_hat)

        n_batch = len(T)

        # Efficient risk set matrix construction
        train_R = (T.view(-1, 1) <= T.view(1, -1)).float().to(device)

        # Flip E: now 0 = event, 1 = censored => event_mask = (E == 0)
        train_ystatus = (E == 0).float().to(device)

        theta = y_hat.reshape(-1)
        exp_theta = torch.exp(theta)

        loss_nn = - torch.mean(
            (theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus
        )

        return loss_nn

##############################################
# General Loss for Discriminator and Generator
##############################################
def real_fake_loss(real, fake, which='bce'):
    fake = fake.squeeze()
    if which == 'bce':
        fake = torch.sigmoid(fake)
        loss = - torch.mean(1.0 - torch.log(fake + 1e-8))
        if real is not None:
            real = real.squeeze()
            real = torch.sigmoid(real)
            loss = loss - torch.mean(torch.log(real + 1e-8))
    elif which == 'hinge':
        loss = nn.ReLU()(1.0 + fake).mean()
        if real is not None:
            real = real.squeeze()
            loss = loss + nn.ReLU()(1.0 - real).mean()
    elif which == 'wasserstein':
        loss = fake.mean()
        if real is not None:
            real = real.squeeze()
            loss = loss - real.mean()
    else:
        loss = None
    return loss


def fake_generator_loss(fake_score):
    # using the value before applying sigmoid -> fake = sigmoid(fake_score)
    fake_score = fake_score.squeeze()
    return - torch.mean(fake_score)


# TODO: sinkhorn survival loss
class SinkhornSurvLoss(nn.Module):
    def __init__(self, alpha=0.0, beta=1, eps=1e-7, reduction='sum', buffer_size = 8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction
        self.beta = beta
        self.buffer_size = buffer_size
        self.h_dict = {"h": [], "t": [], "c": []} # Rolling buffer storage for training data

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        nll_loss_value = nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)

        # Add new batch and maintain buffer size
        batch_size = h.size(0)
        if batch_size == 1:
            if len(self.h_dict["h"]) >= self.buffer_size:
                # Pop oldest sample to maintain buffer size
                self.h_dict["h"].pop(0)
                self.h_dict["t"].pop(0)
                self.h_dict["c"].pop(0)
        else:
            # pop the first batch
            self.h_dict["h"] = self.h_dict["h"][batch_size:]
            self.h_dict["t"] = self.h_dict["t"][batch_size:]
            self.h_dict["c"] = self.h_dict["c"][batch_size:]


        # Add new sample
        self.h_dict["h"].append(h.detach())  # Detach to prevent computation graph growth
        self.h_dict["t"].append(t.detach())
        self.h_dict["c"].append(c.detach())

        # Compute Sinkhorn loss only if buffer is full
        if len(self.h_dict["h"]) < 2:
            return nll_loss_value  # Return only NLL loss until buffer is full

        # Stack tensors for loss computation
        h_stacked = torch.cat(self.h_dict["h"], dim=0)
        t_stacked = torch.cat(self.h_dict["t"], dim=0)
        c_stacked = torch.cat(self.h_dict["c"], dim=0)

        # Compute Sinkhorn loss
        sinkhorn_loss_value = survival_sinkhorn_loss(h_stacked, t_stacked, c_stacked)

        # Return total loss
        return nll_loss_value + self.beta * sinkhorn_loss_value


def survival_sinkhorn_loss(h,t,c):
    """
        Compute Sinkhorn ranking loss for survival prediction.

        Parameters
        ----------
        h: (n_batches, n_classes)
            Neural network output.
        t: (n_batches, 1)
            True survival times.
        c: (n_batches, 1)
            Censoring status.
    """

    hazards = torch.sigmoid(h)  # h: the output of the model
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1)

    # Compute the pairwise risk differences matrix
    S = -(risk.view(-1, 1) - risk.view(1, -1)) # S_ij = risk_i - risk_j

    # # Censor mask: only compare individuals with uncensored survival times
    # valid_pairs = ((c.view(-1, 1) == 0) | (c.view(1, -1) == 0)).float()
    #
    # # Mask out comparisons where T_j < T_i is j is censored
    # consor_mask = ((t.view(-1, 1) >= t.view(1, -1))| (c.view(1, -1) == 0)).float()
    # valid_pairs = valid_pairs * consor_mask

    # Compute approximate ranking matrix using Gumbel-Sinkhorn algorithm
    P = gumbel_sinkhorn_ranking(S)

    # generate ground truth ranking matrix
    true_rank = get_permutation_matrix(t,c)

    # Compute the Sinkhorn loss
    loss = torch.mean(((P - true_rank)**2))
    return loss



def sample_gumbel(shape, eps=1e-20):
    """ Generate Gumbel noise """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sinkhorn_ranking(S, tau=0.1, n_iters=30):
    """ Gumbel-Sinkhorn for survival ranking """
    G = sample_gumbel(S.shape).to(S.device)
    S = S + G  # Differentiable sampling
    P = torch.exp(-S / tau)  # Softmax approximation
    for _ in range(n_iters):
        P = P / P.sum(dim=-1, keepdim=True)  # Row normalization
        P = P / P.sum(dim=-2, keepdim=True)  # Column normalization
    return P

def get_permutation_matrix(t,c):
    """
    Compute the permutation matrix for the true ranking matrix
    """
    sorted_indices = torch.argsort(c*1e6+t,dim=0)
    num_samples = t.shape[0]
    P = torch.zeros((num_samples,num_samples)).to(t.device)
    P[torch.arange(num_samples),sorted_indices] = 1
    return P

if __name__ == "__main__":
    t = torch.tensor([2,1,3,4,5,6,7,8,9,10]).float()
    c = torch.tensor([1,0,0,0,0,0,0,0,0,0]).float()
    p = get_permutation_matrix(t,c)
    print(p)


##################################################
# Unified Survival Objective (per-event NLL + pairwise
# margin-calibrated ranking penalty in one module).
##################################################

class UnifiedSurvivalObjective(nn.Module):
    """统一生存目标：把 per-event NLL 生存损失与 Cox 式成对排序约束
    （margin-calibrated concordance penalty）整合为单个模块的单个标量输出。

    设计上不是发明新的似然公式，而是复用现有实现的逐样本计算逻辑：
    - `_per_event_nll` 复用 `OTEHV2RankEvent._nll_per_sample` 的逐样本 NLL 计算逻辑
      （`survot_rank/research/methods/prognostic_event_transport/model.py`）。
    - `_pairwise_margin_penalty` 复用 `OTEHV2RankEvent._ranking_loss` 的可比对筛选逻辑
      （`comparable = (e>0.5) & (ti<tj)`）以及 `softplus` margin 惩罚项。

    Parameters
    ----------
    margin: float
        Cox 排序惩罚项的 margin，默认为 0.0（与 `OTEHV2RankEvent.rank_margin` 默认值一致）。
    rank_weight: float
        排序惩罚项相对 NLL 的权重系数。
    eps: float
        数值稳定常数，避免对极小值取 log。
    """

    def __init__(self, margin=0.0, rank_weight=1.0, eps=1e-7):
        super().__init__()
        self.margin = margin
        self.rank_weight = rank_weight
        self.eps = eps

    @staticmethod
    def _nll_per_sample(logits, y, c, eps=1e-7):
        """逐样本离散时间生存 NLL。

        与 `OTEHV2RankEvent._nll_per_sample` 逐位一致，独立复制到此处以便
        `loss_func.py` 内部可单独测试，不依赖 `prognostic_event_transport` 模块。
        """
        y = y.view(-1, 1).long()
        c = c.view(-1, 1).float()
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        survival_pad = torch.cat([torch.ones_like(c), survival], dim=1)
        s_prev = torch.gather(survival_pad, 1, y).clamp_min(eps)
        h_this = torch.gather(hazards, 1, y).clamp_min(eps)
        s_this = torch.gather(survival_pad, 1, y + 1).clamp_min(eps)
        uncensored = -(1.0 - c) * (torch.log(s_prev) + torch.log(h_this))
        censored = -c * torch.log(s_this)
        return (uncensored + censored).view(-1)

    def _per_event_nll(self, event_logits, y, c):
        """对 event_logits 计算逐样本 NLL 后取均值。

        - 若 `event_logits` 形状为 `[batch, num_events, num_classes]`（即
          `OTEHV2RankEvent._per_event_surv_loss` 的输入形状），展平为
          `[batch*num_events, num_classes]` 后逐样本计算再取均值。
        - 若 `event_logits` 形状为 `[batch, num_classes]`，直接逐样本计算后取均值。
        """
        if event_logits.dim() == 3:
            bsz, num_events, num_classes = event_logits.shape
            flat = event_logits.reshape(bsz * num_events, num_classes)
            y_rep = y.view(-1, 1).expand(-1, num_events).reshape(-1)
            c_rep = c.view(-1, 1).expand(-1, num_events).reshape(-1)
            per_sample = self._nll_per_sample(flat, y_rep, c_rep, eps=self.eps)
        else:
            per_sample = self._nll_per_sample(event_logits, y, c, eps=self.eps)
        return per_sample.mean()

    @staticmethod
    def _risk_from_logits(logits):
        """由离散时间 hazard logits 计算标量风险分数（越大风险越高）。"""
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        risk = -survival.sum(dim=1)
        return risk

    def _pairwise_margin_penalty(self, risk, y, c):
        """成对 margin 惩罚项，复用 `OTEHV2RankEvent._ranking_loss` 的可比对筛选逻辑。

        `comparable = (e>0.5) & (ti<tj)`：仅当样本 i 未删失且 t_i < t_j 时，
        (i, j) 才被视为一对可比对（Cox 意义下 i 的风险理应大于 j）。
        无可比对时返回 `risk.sum()*0.0`，保持梯度图连通、值为 0。
        """
        t = y.float().view(-1)
        e = (1.0 - c.float()).view(-1)
        if risk.numel() < 2 or e.sum() <= 0:
            return risk.sum() * 0.0
        ti = t.view(-1, 1)
        tj = t.view(1, -1)
        comparable = (e.view(-1, 1) > 0.5) & (ti < tj)
        if comparable.sum() == 0:
            return risk.sum() * 0.0
        diff = risk.view(-1, 1) - risk.view(1, -1)
        values = F.softplus(-(diff - self.margin))[comparable]
        return values.mean()

    def forward(self, event_logits=None, risk_logits=None, y=None, c=None):
        """计算统一生存目标标量损失。

        Parameters
        ----------
        event_logits: (batch, num_events, num_classes) 或 (batch, num_classes)，可选
            事件级 hazard logits，用于计算 per-event NLL；若未提供 `risk_logits`，
            也会作为风险分数的来源（3 维时对 `num_events` 维取均值）。
        risk_logits: (batch, num_classes)，可选
            用于计算风险分数与排序惩罚项的 hazard logits。
        y: (batch,) 或 (batch, 1)
            离散时间标签（时间箱索引）。
        c: (batch,) 或 (batch, 1)
            删失指示（1 表示删失，0 表示观测到事件）。

        Returns
        -------
        Tensor
            0 维（标量）张量，dtype 与输入 logits 一致。
        """
        if event_logits is None and risk_logits is None:
            raise ValueError(
                "UnifiedSurvivalObjective 需要至少提供 event_logits 或 risk_logits 之一"
            )

        ref_logits = event_logits if event_logits is not None else risk_logits
        dtype = ref_logits.dtype
        device = ref_logits.device

        base_nll = torch.zeros((), dtype=dtype, device=device)
        if event_logits is not None:
            base_nll = self._per_event_nll(event_logits, y, c)

        if risk_logits is not None:
            risk_source = risk_logits
        elif event_logits.dim() == 3:
            risk_source = event_logits.mean(dim=1)
        else:
            risk_source = event_logits
        risk = self._risk_from_logits(risk_source)

        concordance_penalty = self._pairwise_margin_penalty(risk, y, c)

        total = base_nll + self.rank_weight * concordance_penalty
        return total.to(dtype)
