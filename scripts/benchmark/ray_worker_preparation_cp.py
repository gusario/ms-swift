# Copyright (c) ModelScope Contributors. All rights reserved.
"""GPU parity of native and worker-prepared GRPO inputs; launch with RAY_AUTH_MODE=token and torchrun."""
import argparse
import copy
import json
import os
import ray
import tempfile
import torch
import torch.distributed as dist
import traceback
from dataclasses import MISSING, fields
from pathlib import Path
from ray_worker_preparation import make_group, make_samples, make_template, make_trainer, signature
from types import SimpleNamespace

from swift.ray.megatron.base_trainer import BaseRayTrainer
from swift.ray.megatron.data import resolve_batches
from swift.ray.megatron.megatron_worker import MegatronWorker
from swift.rl_core.advantage import compute_advantages


@ray.remote(num_cpus=0)
class BatchStore:

    def __init__(self):
        self.batches = {}

    def set(self, label, dispatch):
        self.batches[label] = dispatch

    def get(self, label, rank):
        return self.batches[label][rank]


def make_args(cp, dp, padding_free):
    from swift.megatron.arguments import MegatronRLHFArguments

    defaults = {}
    for field in fields(MegatronRLHFArguments):
        if field.default is not MISSING:
            defaults[field.name] = copy.deepcopy(field.default)
        elif field.default_factory is not MISSING:
            defaults[field.name] = field.default_factory()
    defaults.update(
        context_parallel_size=cp,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        cp_partition_mode='zigzag',
        is_multimodal=False,
        padding_free=padding_free,
        task_type='causal_lm',
        micro_batch_size=2,
        loss_type='dapo',
        num_microbatches=2,
        num_generations=4,
        global_batch_size=4 * dp,
        generation_batch_size=4 * dp,
        steps_per_generation=1,
        per_device_generation_batch_size=4,
        max_completion_length=32,
        beta=0.0,
        temperature=0.7,
        kl_in_reward=False,
        rollout_importance_sampling_mode=None,
        log_rollout_offpolicy_metrics=False,
        router_replay_mode='disabled',
        report_to=[],
        overlong_filter=True)
    return SimpleNamespace(**defaults)


def main():
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    from swift.megatron.trainers.utils import compute_per_token_logps_fn
    from swift.ray.megatron.loss.grpo import GRPOLoss
    from swift.utils import to_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cp', type=int, choices=[1, 2], default=2)
    parser.add_argument('--cp-comm', choices=['p2p', 'a2a'], default='p2p')
    parser.add_argument('--padding-free', action='store_true')
    options = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(1)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if world % options.cp:
        raise ValueError('WORLD_SIZE must be divisible by --cp')
    group, temp_dir, loss_dir, store_handle = None, None, None, [None]
    try:
        mpu.initialize_model_parallel(context_parallel_size=options.cp)
        dp, dp_rank = mpu.get_data_parallel_world_size(), mpu.get_data_parallel_rank()
        args = make_args(options.cp, dp, options.padding_free)
        loss_dir = tempfile.TemporaryDirectory(prefix='swift-pr9loss-', dir='/tmp')
        args.output_dir = loss_dir.name
        os.environ['RAY_AUTH_MODE'] = 'token'
        flag = 'RAY_ENABLE_ZERO_COPY_TORCH_TENSORS'
        script_dir = str(Path(__file__).resolve().parent)
        env = {
            flag: os.environ.get(flag, '0'),
            'PYTHONPATH': script_dir + os.pathsep + os.environ.get('PYTHONPATH', '')
        }
        address = [None]
        if rank == 0:
            temp_dir = tempfile.TemporaryDirectory(prefix='swift-pr9cp-', dir='/tmp')
            ray.init(
                address='local',
                num_cpus=world,
                num_gpus=0,
                include_dashboard=False,
                object_store_memory=256 << 20,
                _temp_dir=temp_dir.name,
                runtime_env={'env_vars': env})
            address[0] = ray.get_runtime_context().gcs_address
        dist.broadcast_object_list(address, src=0)
        if rank != 0:
            ray.init(address=address[0], runtime_env={'env_vars': env})

        store_handle = [None]
        if rank == 0:
            store = BatchStore.remote()
            store_handle[0] = ray.cloudpickle.dumps(store)
        dist.broadcast_object_list(store_handle, src=0)
        store = ray.cloudpickle.loads(store_handle[0])
        if rank == 0:
            template = make_template(options.cp, options.padding_free)
            group = make_group(args, template, dp, options.cp)
            trainer = make_trainer(args, template, group)
            samples = make_samples(4 * dp, repeats=1)
            native_samples = copy.deepcopy(samples)
            trainer.encode_rollout_batch(native_samples)
            native, native_batches = BaseRayTrainer._collate_for_workers(trainer, group, native_samples)
            # Exercise the actual operation below its production DP>8 activation threshold.
            prepared, prepared_batches = trainer._prepare_rollout_on_workers(group, copy.deepcopy(samples))
            for index in range(dp):
                assert signature(resolve_batches(prepared[index])) == signature(native[index]), index
            print('CHECK native/prepared CPU inputs and metadata match', flush=True)
            for label, dispatch in [('driver', native), ('worker', prepared)]:
                ray.get(store.set.remote(label, dispatch))
        dist.barrier()

        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        cfg = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=2,
            num_query_groups=2,
            ffn_hidden_size=256,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            params_dtype=torch.bfloat16,
            bf16=True,
            perform_initialization=True,
            attention_backend=AttnBackend.flash,
            normalization='RMSNorm',
            gated_linear_unit=True,
            add_bias_linear=False,
            context_parallel_size=options.cp,
            cp_comm_type=options.cp_comm,
            gradient_accumulation_fusion=False)
        model = GPTModel(
            cfg,
            get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=512,
            max_sequence_length=2048,
            position_embedding_type='rope',
            parallel_output=True,
            share_embeddings_and_output_weights=False).cuda()
        with torch.no_grad():
            for parameter in model.parameters():
                dist.broadcast(parameter, src=0)
        worker = MegatronWorker()
        worker._args = args
        worker._pipeline = SimpleNamespace(template=SimpleNamespace(padding_side='right'))
        worker._megatron = SimpleNamespace(
            unwrapped_models=[model],
            enable_routing_replay=False,
            compute_per_token_logps=lambda model, iterator, temperature: compute_per_token_logps_fn(
                model, args, iterator, temperature=temperature))
        worker.ref = SimpleNamespace(models=[model])
        if rank == 0:
            print('CHECK native CP model initialized', flush=True)
        scores = {}
        for label in ['driver', 'worker']:
            payload = ray.get(store.get.remote(label, dp_rank))
            scores[label] = worker.compute_logps(payload)
            reference = worker.compute_ref_logps(ray.get(store.get.remote(label, dp_rank)))
            for actor_row, ref_row in zip(scores[label], reference):
                torch.testing.assert_close(actor_row['per_token_logps'], ref_row['ref_per_token_logps'], rtol=0, atol=0)
        max_logp_error = 0.0
        for expected, actual in zip(scores['driver'], scores['worker']):
            torch.testing.assert_close(actual['per_token_logps'], expected['per_token_logps'], atol=2e-5, rtol=1e-6)
            max_logp_error = max(max_logp_error,
                                 (actual['per_token_logps'] - expected['per_token_logps']).abs().max().item())
        if rank == 0:
            print('CHECK native/prepared actor and reference scores match', flush=True)
        all_scores = [None] * world
        dist.all_gather_object(all_scores, (worker.get_parallel_info(), scores['driver']))
        if rank == 0:
            ordered = {info['dp_rank']: rows for info, rows in all_scores if info['is_collector']}
            rows = [row for index in range(dp) for row in ordered[index]]
            rewards = torch.arange(4 * dp, dtype=torch.float32).remainder(5)[:, None]
            advantages, _ = compute_advantages(rewards, torch.ones(1), 4, 'grpo', 'group')
            for label, dispatch, batches in [('driver', native, native_batches),
                                             ('worker', prepared, prepared_batches)]:
                trainer._scatter_logps(batches, rows, 'old_per_token_logps')
                trainer._scatter_advantages(batches, advantages)
                for batch in batches:
                    # Both paths use the declared tensor normalizer; the original driver returns an int.
                    batch.num_items_in_batch = torch.as_tensor(batch.num_items_in_batch)
                ray.get(store.set.remote(label, dispatch))
        dist.barrier()

        initial = copy.deepcopy(model.state_dict())
        snapshots = {}
        for label in ['driver', 'worker']:
            model.load_state_dict(initial)
            model.train()
            model.zero_grad(set_to_none=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
            criterion = GRPOLoss(args)
            criterion._dummy._has_teacher = False
            losses = []
            for inputs in resolve_batches(ray.get(store.get.remote(label, dp_rank))):
                batch = inputs.pop('grpo_batch').to_device(torch.cuda.current_device())
                inputs = to_device(inputs, torch.cuda.current_device())
                inputs['grpo_batch'] = batch
                output, loss_func = criterion.forward_step(iter([inputs]), model)
                loss, _ = loss_func(output)
                (loss / args.num_microbatches).backward()
                losses.append(loss.detach())
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, group=mpu.get_data_parallel_group(with_context_parallel=True))
                    # CP contributes disjoint token gradients; average only across DP.
                    parameter.grad.div_(dp)
                    gradients[name] = parameter.grad.detach().clone()
            assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
            assert any(value.ne(0).any() for value in gradients.values())
            optimizer.step()
            assert any(not torch.equal(value, initial[key]) for key, value in model.named_parameters())
            snapshots[label] = {
                'losses': torch.stack(losses),
                'gradients': gradients,
                'parameters': {
                    key: value.detach().clone()
                    for key, value in model.named_parameters()
                },
                'moments': {
                    f'{name}/{key}': value.detach().clone()
                    for name, parameter in model.named_parameters()
                    for key, value in optimizer.state[parameter].items()
                }
            }
        max_errors = {'logps': max_logp_error}
        for field in ['losses', 'gradients', 'parameters', 'moments']:
            expected, actual = snapshots['driver'][field], snapshots['worker'][field]
            if isinstance(expected, torch.Tensor):
                expected, actual = {'loss': expected}, {'loss': actual}
            error = 0.0
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=1e-6)
                error = max(error, (actual[key] - expected[key]).abs().max().item())
            max_errors[field] = error
        results = [None] * world
        dist.all_gather_object(results, max_errors)
        if rank == 0:
            print(
                'RESULT ' + json.dumps({
                    'world_size': world,
                    'dp': dp,
                    'cp': options.cp,
                    'cp_comm': options.cp_comm,
                    'padding_free': options.padding_free,
                    'forced_worker_operation': True,
                    'native_inputs_metadata_order_equal': True,
                    'native_logps_loss_gradients_adamw_equal': True,
                    'rank_max_errors': results
                }),
                flush=True)
        dist.barrier()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if rank == 0 and group is not None:
            group.shutdown(timeout=5)
        if rank == 0 and ray.is_initialized() and store_handle[0] is not None:
            ray.kill(store, no_restart=True)
        ray.shutdown()
        mpu.destroy_model_parallel()
        dist.destroy_process_group()
        if temp_dir is not None:
            temp_dir.cleanup()
        if loss_dir is not None:
            loss_dir.cleanup()


if __name__ == '__main__':
    main()
