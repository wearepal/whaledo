from typing import Callable, Optional, Type, TypeVar, Union, cast

import torch
from torch import Tensor
from torch.autograd.function import Function, NestedIOFunction
import torch.distributed
import torch.nn as nn
from typing_extensions import Self

__all__ = [
    "DecoupledContrastiveLoss",
    "decoupled_contrastive_loss",
    "moco_v2_loss",
    "simclr_loss",
    "supcon_loss",
]


class _Synchronize(Function):
    @staticmethod
    def forward(ctx: NestedIOFunction, tensor: Tensor) -> Tensor:
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: NestedIOFunction, grad_output: Tensor) -> Tensor:
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


def maybe_synchronize(input: Tensor) -> Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return _Synchronize.apply(input)
    return input


def logsumexp(
    input: Tensor, *, dim: int, keepdim: bool = False, keep_mask: Optional[Tensor] = None
) -> Tensor:
    """Numerically stable logsumexp on the last dim of `inputs`.
    reference: https://github.com/pytorch/pytorch/issues/2591
    """
    if keep_mask is None:
        return input.logsumexp(dim=dim, keepdim=keepdim)
    eps = torch.finfo(input.dtype).eps
    max_offset = eps * keep_mask.to(input.dtype)
    max_ = torch.max(input + max_offset, dim=dim, keepdim=True).values
    input = input - max_
    if keepdim is False:
        max_ = max_.squeeze(dim)
    input_exp_m = input.exp() * keep_mask
    return max_ + input_exp_m.sum(dim=dim, keepdim=keepdim).log()


def moco_v2_loss(
    anchors: Tensor,
    *,
    positives: Tensor,
    negatives: Tensor,
    temperature: Union[float, Tensor] = 1.0,
    dcl: bool = True,
) -> Tensor:
    positives = maybe_synchronize(positives)
    negatives = maybe_synchronize(negatives)
    if (positives.requires_grad) or (negatives.requires_grad):
        anchors = maybe_synchronize(anchors)

    n, d = anchors.size(0), anchors.size(-1)
    anchors = anchors.view(n, -1, d)
    positives = positives.view(n, 1, d)
    l_pos = (anchors * positives).sum(-1).view(-1, 1) / temperature
    l_neg = (anchors @ negatives.T).view(l_pos.size(0), -1) / temperature
    # Compute the partition function either according to the original InfoNCE formulation
    # or according to the DCL formulation which excludes the positive samples.
    z = l_neg.logsumexp(dim=1) if dcl else torch.cat([l_pos, l_neg], dim=1).logsumexp(dim=1)
    return (z - l_pos).mean()


def simclr_loss(
    anchors: Tensor,
    *,
    targets: Tensor,
    temperature: Union[float, Tensor] = 1.0,
    dcl: bool = True,
) -> Tensor:
    anchors = maybe_synchronize(anchors)
    targets = maybe_synchronize(targets)

    logits = (anchors @ targets.T) / temperature
    pos_idxs = torch.arange(logits.size(0)).view(-1, *((1,) * (logits.ndim - 1)))
    l_pos = logits.gather(-1, pos_idxs)

    z_mask = None
    if dcl:
        z_mask = ~torch.eye(len(logits), dtype=torch.bool, device=logits.device)
        if anchors.ndim == 3:
            z_mask = z_mask.unsqueeze(1)
    z = logsumexp(logits, dim=-1, keep_mask=z_mask)
    return (z.sum() - l_pos.sum()) / z.numel()


T = TypeVar("T", Tensor, None)


def supcon_loss(
    anchors: Tensor,
    *,
    anchor_labels: Tensor,
    candidates: T = None,
    candidate_labels: T = None,
    temperature: Union[float, Tensor] = 0.1,
    exclude_diagonal: bool = False,
    dcl: bool = True,
) -> Tensor:
    if len(anchors) != len(anchor_labels):
        raise ValueError("'anchors' and 'anchor_labels' must match in size at dimension 0.")
    # Create new variables for the candidate- variables to placate
    # the static-type checker.

    if candidates is None:
        candidates_t = anchors
        candidate_labels_t = anchor_labels
        # Forbid interactions between the samples and themsleves.
        exclude_diagonal = True
    else:
        candidates_t = candidates
        candidate_labels_t = cast(Tensor, candidate_labels)
        if len(candidates_t) != len(candidate_labels_t):
            raise ValueError(
                "'candidates' and 'candidate_labels' must match in size at dimension 0."
            )
        candidates_t = maybe_synchronize(candidates_t)
        candidate_labels_t = maybe_synchronize(candidate_labels_t)
        # If the gradient is to be computed bi-directionally then both the queries and the keys
        # need to be collated across all devices (otherwise just doing so for the keys is
        # sufficient).
        if candidates_t.requires_grad:
            anchors = maybe_synchronize(anchors)
            anchor_labels = maybe_synchronize(anchor_labels)

    anchor_labels = anchor_labels.view(-1, 1)
    candidate_labels_t = candidate_labels_t.flatten()
    # The positive samples for a given anchor are those samples from the candidate set sharing its
    # label.
    pos_mask = anchor_labels == candidate_labels_t
    neg_mask = None
    if dcl:
        neg_mask = ~pos_mask
    elif exclude_diagonal:
        neg_mask = ~torch.eye(len(pos_mask), dtype=torch.bool, device=pos_mask.device)
    if exclude_diagonal:
        pos_mask.fill_diagonal_(False)

    pos_inds = pos_mask.nonzero(as_tuple=True)
    row_inds, col_inds = pos_inds
    # Return early if there are no positive pairs.
    if len(row_inds) == 0:
        return anchors.new_zeros(())
    # Only compute the pairwise similarity for those samples which have positive pairs.
    selected_rows, row_inverse, row_counts = row_inds.unique(
        return_inverse=True, return_counts=True
    )
    logits = anchors[selected_rows] @ candidates_t.T
    # Apply temperature-scaling to the logits.
    logits = logits / temperature
    # Subtract the maximum for numerical stability.
    logits_max = logits.max(dim=1, keepdim=True).values
    logits = logits - logits_max.detach()
    # Tile the row counts if dealing with multicropping.
    if anchors.ndim == 3:
        row_counts = row_counts.unsqueeze(1).expand(-1, anchors.size(1))
    counts_flat = row_counts[row_inverse].flatten()
    positives = logits[row_inverse, ..., col_inds].flatten() / counts_flat

    if neg_mask is not None:
        neg_mask = neg_mask[selected_rows]
        if anchors.ndim == 3:
            neg_mask = neg_mask.unsqueeze(1)
    z = logsumexp(logits, dim=-1, keep_mask=neg_mask)
    return (z.sum() - positives.sum()) / z.numel()


def decoupled_contrastive_loss(
    z1: Tensor,
    z2: Tensor,
    *,
    temperature: float = 0.1,
    weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> Tensor:
    """
    Calculates the one-way `decoupled constrastive loss <https://arxiv.org/pdf/2110.06848.pdf>`_.

    :param z1: First embedding vector
    :param z2: Second embedding vector
    :param weight_fn: The weighting function of the positive sample loss.
    :param temperature: Temperature controlling the sharpness of the softmax distribution.

    :return: One-way loss between the embedding vectors.
    """
    cross_view_distance = torch.mm(z1, z2.t())
    positive_loss = -torch.diag(cross_view_distance) / temperature
    if weight_fn is not None:
        positive_loss = positive_loss * weight_fn(z1, z2)
    neg_similarity = torch.cat((z1 @ z1.t(), cross_view_distance), dim=1) / temperature
    neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
    eps = torch.finfo(z1.dtype).eps
    negative_loss = torch.logsumexp(neg_similarity + neg_mask * eps, dim=1, keepdim=False)
    return (positive_loss + negative_loss).mean()


def _von_mises_fisher_weighting(z1: Tensor, z2: Tensor, *, sigma: float = 0.5) -> Tensor:
    return 2 - len(z1) * ((z1 * z2).sum(dim=1) / sigma).softmax(dim=0).squeeze()


class DecoupledContrastiveLoss(nn.Module):
    """
    Implementation of the Decoupled Contrastive Loss proposed in
    https://arxiv.org/pdf/2110.06848.pdf, adapted from the `official code
    <https://github.com/raminnakhli/Decoupled-Contrastive-Learning>`_
    """

    def __init__(
        self,
        temperature: float = 0.1,
        *,
        weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        """
        :param weight_fn: The weighting function of the positive sample loss.
        :param temperature: Temperature controlling the sharpness of the softmax distribution.
        """
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        Calculates the one-way decoupled constrastive loss.

        :param z1: First embedding vector
        :param z2: Second embedding vector
        :return: One-way loss between the embedding vectors.
        """
        return decoupled_contrastive_loss(
            z1, z2, temperature=self.temperature, weight_fn=self.weight_fn
        )

    @classmethod
    def with_vmf_weighting(
        cls: Type[Self], sigma: float = 0.5, *, temperature: float = 0.1
    ) -> Self:
        """
        Initialise the DCL loss with von Mises-Fisher weighting.

        :param sigma: :math:`\\sigma` (scale) parameter for the weigting function.
        :param temperature: Temperature controlling the sharpness of the softmax distribution.
        """
        return cls(temperature=temperature, weight_fn=_von_mises_fisher_weighting)
