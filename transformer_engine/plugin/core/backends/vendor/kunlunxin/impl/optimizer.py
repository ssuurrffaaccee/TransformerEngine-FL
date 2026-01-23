# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Union
import torch

__all__ = [
    "multi_tensor_adam_torch"
]

def multi_tensor_adam_torch(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
) -> None:
    if noop_flag.item() != 0:
        return

    if len(tensor_lists) != 4:
        raise ValueError("tensor_lists should contain [grads, params, exp_avgs, exp_avg_sqs]")

    grads, params, exp_avgs, exp_avg_sqs = tensor_lists

    if not (len(params) == len(grads) == len(exp_avgs) == len(exp_avg_sqs)):
        raise ValueError("All tensor lists must have the same length")

    if bias_correction:
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
    else:
        bias_correction1 = 1.0
        bias_correction2 = 1.0

    for grad, param, exp_avg, exp_avg_sq in zip(grads, params, exp_avgs, exp_avg_sqs):
        if grad is None:
            continue

        if mode == 1 and weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        corrected_exp_avg = exp_avg / bias_correction1
        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

        denom = corrected_exp_avg_sq.sqrt().add_(eps)
        param.addcdiv_(corrected_exp_avg, denom, value=-lr)
