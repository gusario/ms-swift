# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU collation and object-store inputs shared by Ray drivers and workers."""
from __future__ import annotations

import copy
import ray
from dataclasses import dataclass
from typing import Any, Dict, List

from swift.rl_core.data import GRPOBatch
from .worker_group import DPDispatchedDict


@dataclass
class PreparedBatch:
    """Object-store model inputs and current driver metadata for one micro-batch."""

    model_inputs: ray.ObjectRef
    grpo_batch: GRPOBatch


def resolve_batches(micro_batches: List[Any]) -> List[Dict[str, Any]]:
    refs = [batch.model_inputs for batch in micro_batches if isinstance(batch, PreparedBatch)]
    resolved = iter(ray.get(refs) if refs else [])
    result = []
    for batch in micro_batches:
        if isinstance(batch, PreparedBatch):
            inputs = dict(next(resolved))
            # Scores and advantages arrive after preparation; consumers mutate metadata.
            inputs['grpo_batch'] = copy.copy(batch.grpo_batch)
            result.append(inputs)
        else:
            result.append(batch)
    return result


def collate_rollout_batch(samples, args, template, dp_size, device, padding_to, **collate_kwargs):
    from swift.rlhf_trainers.utils import collate_to_grpo_micro_batch

    mbs = int(args.micro_batch_size)
    n = len(samples)
    if n % dp_size != 0:
        raise ValueError(f'_collate_for_workers: batch size {n} not divisible by dp_size {dp_size}.')
    shard_size = n // dp_size

    dispatch = DPDispatchedDict()
    flat_grpo_batches: List['GRPOBatch'] = []
    for dp_rank in range(dp_size):
        shard = samples[dp_rank * shard_size:(dp_rank + 1) * shard_size]
        micro_batches = []
        for i in range(0, len(shard), mbs):
            chunk = shard[i:i + mbs]
            model_inputs, grpo_batch = collate_to_grpo_micro_batch(
                chunk,
                template,
                device=device,
                padding_to=padding_to,
                router_replay_mode=getattr(args, 'router_replay_mode', 'disabled'),
                **collate_kwargs,
            )
            model_inputs['grpo_batch'] = grpo_batch
            micro_batches.append(model_inputs)
            flat_grpo_batches.append(grpo_batch)
        dispatch[dp_rank] = micro_batches
    if getattr(args, 'loss_type', None) in ['bnpo', 'cispo', 'dapo', 'fipo']:
        num_items_in_batch = sum(grpo_batch.completion_mask.sum().item() for grpo_batch in flat_grpo_batches)
        for grpo_batch in flat_grpo_batches:
            grpo_batch.num_items_in_batch = num_items_in_batch
    return dispatch, flat_grpo_batches
