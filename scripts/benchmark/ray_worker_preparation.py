# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline native GRPO preparation replay using a synthetic Qwen byte-tokenizer fixture."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import ray
import statistics
import sys
import tempfile
import time
import torch
from dataclasses import asdict
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from types import SimpleNamespace

from swift.ray.megatron.data import resolve_batches
from swift.ray.megatron.grpo_trainer import GRPOTrainer
from swift.ray.megatron.megatron_worker import MegatronWorker
from swift.ray.megatron.worker_group import WorkerGroup, dispatch_collect
from swift.rl_core.data import GRPOBatch, GRPOSample
from swift.template import get_template


def make_template(cp=1, padding_free=False):
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token='<|im_end|>',
        pad_token='<|endoftext|>',
        additional_special_tokens=['<|im_start|>', '<|im_end|>'])
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        model_info=SimpleNamespace(config={}, task_type='causal_lm', max_model_len=32768),
        model_meta=SimpleNamespace(is_multimodal=False))
    template = get_template(
        processor, template_type='qwen2_5', padding_free=padding_free, sequence_parallel_size=cp, padding_side='right')
    template.set_mode('train')
    template.use_megatron = True
    return template


def make_samples(count, repeats=8):
    return [
        GRPOSample(
            request_id=str(i),
            messages=[{
                'role': 'user',
                'content': f'Prompt {i // 4}. ' + 'Read the controls and choose the requested action. ' * repeats
            }, {
                'role': 'assistant',
                'content': f'answer {i}'
            }]) for i in range(count)
    ]


def signature(value):
    if isinstance(value, GRPOBatch):
        value = asdict(value)
        if value['num_items_in_batch'] is not None:
            value['num_items_in_batch'] = int(value['num_items_in_batch'])
    if isinstance(value, torch.Tensor):
        data = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return str(value.dtype), tuple(value.shape), hashlib.sha256(data).hexdigest()
    if isinstance(value, dict):
        return {key: signature(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [signature(item) for item in value]
    return value


class PreparationWorker(MegatronWorker):
    """Model-free actor executing the native preparation and input-resolution methods."""

    def __init__(self, args, template, info):
        super().__init__()
        self._args, self._pipeline, self.info = args, SimpleNamespace(template=template), info
        torch.set_num_threads(1)

    def get_parallel_info(self):
        return self.info

    @dispatch_collect(dispatch='dp', collect='dp')
    def consume(self, micro_batches):
        inputs = resolve_batches(micro_batches)
        checksum = sum(int(batch['input_ids'].sum()) for batch in inputs)
        assert all(batch['grpo_batch'].advantages is not None for batch in inputs)
        for batch in inputs:
            batch.pop('grpo_batch')
        return checksum if self.info['is_collector'] else None


def make_group(args, template, dp, cp):
    actor = ray.remote(num_cpus=1)(PreparationWorker)
    workers = [
        actor.remote(args, template, {
            'dp_rank': rank,
            'dp_size': dp,
            'is_collector': peer == 0
        }) for rank in range(dp) for peer in range(cp)
    ]
    group = WorkerGroup('preparation-replay', workers)
    group.build_dispatch_info(PreparationWorker)
    return group


def make_trainer(args, template, group):
    trainer = GRPOTrainer({'train': group}, [])
    trainer.args, trainer.template, trainer.device = args, template, torch.device('cpu')
    trainer._padding_to = 2 * args.context_parallel_size
    trainer._has_teacher = False
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Unmodified upstream base_trainer.py')
    parser.add_argument('--dp', type=int, default=9)
    parser.add_argument('--cp', type=int, choices=[1, 2], default=1)
    parser.add_argument('--padding-free', action='store_true')
    parser.add_argument('--per-rank', type=int, default=4)
    parser.add_argument('--text-repeats', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--object-store-mib', type=int, default=1024)
    options = parser.parse_args()
    if options.dp <= 8 or options.per_rank < 2 or options.per_rank % 2 or min(options.rounds, options.text_repeats) < 1:
        parser.error('Requires DP > 8, positive rounds/repeats, and an even --per-rank >= 2')
    if options.object_store_mib < 80:
        parser.error('--object-store-mib must be >= 80')
    name = 'swift.ray.megatron._preparation_baseline'
    spec = importlib.util.spec_from_file_location(name, options.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    template = make_template(options.cp, options.padding_free)
    args = SimpleNamespace(micro_batch_size=2, context_parallel_size=options.cp, loss_type='dapo')
    samples = make_samples(options.dp * options.per_rank, options.text_repeats)
    timings = {name: [] for name in ['driver', 'worker']}
    group = None
    with tempfile.TemporaryDirectory(prefix='swift-pr9-', dir='/tmp') as temp_dir:
        try:
            flag = 'RAY_ENABLE_ZERO_COPY_TORCH_TENSORS'
            ray.init(
                address='local',
                num_cpus=options.dp * options.cp,
                num_gpus=0,
                include_dashboard=False,
                object_store_memory=options.object_store_mib << 20,
                _temp_dir=temp_dir,
                runtime_env={'env_vars': {
                    flag: os.environ.get(flag, '0')
                }})
            group = make_group(args, template, options.dp, options.cp)
            trainer = make_trainer(args, template, group)

            def run(label):
                batch = copy.deepcopy(samples)
                start = time.perf_counter()
                if label == 'driver':
                    trainer.encode_rollout_batch(batch)
                    dispatch, batches = baseline.BaseRayTrainer._collate_for_workers(trainer, group, batch)
                else:
                    dispatch, batches = trainer._prepare_rollout_batch(group, batch)
                prepared = time.perf_counter() - start
                for item in batches:
                    item.advantages = torch.ones_like(item.completion_mask, dtype=torch.float32)
                start = time.perf_counter()
                first = group.consume(dispatch)
                assert group.consume(dispatch) == first
                assert group.consume(dispatch) == first
                consumed = time.perf_counter() - start
                return dispatch, {
                    'prepare_seconds': prepared,
                    'three_consumers_seconds': consumed,
                    'total_seconds': prepared + consumed
                }, first

            native, _, expected = run('driver')
            prepared, _, actual = run('worker')
            assert actual == expected
            for rank in range(options.dp):
                assert signature(resolve_batches(prepared[rank])) == signature(native[rank]), rank
            del native, prepared
            for index in range(options.rounds):
                for label in (['driver', 'worker'] if index % 2 == 0 else ['worker', 'driver']):
                    dispatch, elapsed, actual = run(label)
                    assert actual == expected
                    timings[label].append(elapsed)
                    del dispatch
            medians = {
                label: {
                    key: statistics.median(row[key] for row in rows)
                    for key in rows[0]
                }
                for label, rows in timings.items()
            }
            print(
                'RESULT ' + json.dumps(
                    {
                        'ray': ray.__version__,
                        'torch': torch.__version__,
                        'driver_torch_threads': torch.get_num_threads(),
                        'dp': options.dp,
                        'cp': options.cp,
                        'padding_free': options.padding_free,
                        'samples': len(samples),
                        'per_rank': options.per_rank,
                        'text_repeats': options.text_repeats,
                        'fixture': 'Native Qwen2.5 template; synthetic byte tokenizer/text',
                        'inputs_metadata_order_equal': True,
                        'warmup_per_variant': 1,
                        'rounds': options.rounds,
                        'seconds': timings,
                        'median_seconds': medians,
                        'prepare_speedup': medians['driver']['prepare_seconds'] / medians['worker']['prepare_seconds'],
                        'total_speedup': medians['driver']['total_seconds'] / medians['worker']['total_seconds']
                    }),
                flush=True)
        finally:
            if group is not None:
                group.shutdown(timeout=5)
            ray.shutdown()


if __name__ == '__main__':
    main()
