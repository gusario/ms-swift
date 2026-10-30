# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import importlib.util
import numpy as np
import random
import torch
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec('ray'), 'Ray is required')
class TestRayWorkerPreparation(unittest.TestCase):

    def setUp(self):
        from swift.ray.megatron.grpo_trainer import GRPOTrainer
        from swift.ray.megatron.megatron_worker import MegatronWorker

        self.trainer_cls, self.worker_cls = GRPOTrainer, MegatronWorker
        self.calls, self.storage = [], {}
        self.args = SimpleNamespace(micro_batch_size=2, loss_type='dapo', context_parallel_size=1)
        self.template = object()
        self.trainer = GRPOTrainer({}, [])
        self.trainer.args, self.trainer.template = self.args, self.template
        self.trainer.device, self.trainer._padding_to = torch.device('cpu'), 4
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.threads)
        for target, value in [
            ('swift.rlhf_trainers.utils.encode_sample', self.encode),
            ('swift.ray.megatron.grpo_trainer.encode_sample', self.encode),
            ('swift.rlhf_trainers.utils.collate_to_grpo_micro_batch', self.collate),
            ('ray.put', self.put),
            ('ray.get', lambda refs: [self.storage[ref] for ref in refs]),
        ]:
            mock = patch(target, side_effect=value)
            mock.start()
            self.addCleanup(mock.stop)

    def samples(self, n=36):
        from swift.rl_core.data import GRPOSample
        return [GRPOSample(messages=[], request_id=str(i)) for i in range(n)]

    def encode(self, sample, template):
        self.assertIs(template, self.template)
        self.calls.append(sample.request_id)
        return {'input_ids': [int(sample.request_id), 7, 8], 'media': [1], 'length': 3, '_extra_kwargs': {}}

    def collate(self, samples, template, **kwargs):
        from swift.rl_core.data import GRPOBatch

        self.assertEqual(kwargs['padding_to'], 4)
        ids = torch.tensor([sample.encoded['input_ids'] for sample in samples])
        lengths = [sample.encoded['length'] for sample in samples]
        for sample in samples:
            self.assertNotIn('_extra_kwargs', sample.encoded)
            sample.encoded.pop('media')
            sample.encoded['length'] = 4
        batch = GRPOBatch(
            torch.tensor([[False, True, True]] * len(samples)), torch.zeros(len(samples), dtype=torch.bool),
            torch.tensor(lengths))
        return {
            'input_ids': ids,
            'model_specific_tensor': ids[:, None].clone(),
            'prepared_inputs': 'a valid native key',
            'metadata': {
                'ids': ids[:, 0].tolist()
            }
        }, batch

    def put(self, value):
        ref = len(self.storage)
        self.storage[ref] = copy.deepcopy(value)
        return ref

    def worker(self, collector=True):
        worker = self.worker_cls()
        worker._args = self.args
        worker._pipeline = SimpleNamespace(template=self.template)
        worker.get_parallel_info = lambda: {'is_collector': collector}
        return worker

    def group(self, dp=9):
        from swift.ray.megatron.worker_group import WorkerGroup

        def prepare(shards, **kwargs):
            results, ranks, collectors = [], [], []
            for rank in range(dp):
                for collector in [False, True]:
                    results.append(self.worker(collector).prepare_rollout_batch(copy.deepcopy(shards[rank]), **kwargs))
                    ranks.append(rank)
                    collectors.append(collector)
            return WorkerGroup._collect_dp(SimpleNamespace(_dp_rank_map=ranks, _collect_mask=collectors), results)

        return SimpleNamespace(dp_size=dp, prepare_rollout_batch=prepare)

    def test_collectors_prepare_each_sample_once_in_native_order(self):
        from swift.ray.megatron.data import PreparedBatch, resolve_batches

        samples = self.samples()
        for sample in samples:
            sample.encoded = {'input_ids': [-1]}  # A preceding rollout's encoding must not be reused.
        dispatch, batches = self.trainer._prepare_rollout_batch(self.group(), samples)
        self.assertEqual(Counter(self.calls), Counter(str(i) for i in range(36)))
        self.assertTrue(all(sample.encoded is None for sample in samples))
        ids = []
        for rank in range(9):
            self.assertEqual(len(dispatch[rank]), 2)
            self.assertTrue(all(isinstance(batch, PreparedBatch) for batch in dispatch[rank]))
            for inputs in resolve_batches(dispatch[rank]):
                ids.extend(inputs['input_ids'][:, 0].tolist())
                self.assertEqual(inputs['prepared_inputs'], 'a valid native key')
                torch.testing.assert_close(inputs['model_specific_tensor'][:, 0], inputs['input_ids'])
        self.assertEqual(ids, list(range(36)))
        self.assertTrue(all(batch.num_items_in_batch.item() == 72 for batch in batches))
        self.assertTrue(all(isinstance(batch.num_items_in_batch, torch.Tensor) for batch in batches))

    def test_native_driver_fallback_for_small_dp_and_partial_shards(self):
        for dp, cp, count in [(1, 1, 4), (8, 1, 32), (9, 1, 27), (9, 1, 0)]:
            with self.subTest(dp=dp, cp=cp, count=count):
                self.args.context_parallel_size = cp
                group = SimpleNamespace(dp_size=dp, prepare_rollout_batch=Mock(side_effect=AssertionError('remote')))
                self.calls.clear()
                dispatch, batches = self.trainer._prepare_rollout_batch(group, self.samples(count))
                self.assertEqual(self.calls, [str(i) for i in range(count)])
                group.prepare_rollout_batch.assert_not_called()
                self.assertEqual(sum(batch.completion_mask.shape[0] for batch in batches), count)
                self.assertTrue(all(isinstance(batch, dict) for shard in dispatch.values() for batch in shard))

    def test_context_parallel_uses_worker_preparation(self):
        from swift.ray.megatron.data import PreparedBatch

        self.args.context_parallel_size = 2
        dispatch, _ = self.trainer._prepare_rollout_batch(self.group(), self.samples())
        self.assertTrue(all(isinstance(batch, PreparedBatch) for shard in dispatch.values() for batch in shard))
        self.assertEqual(len(self.calls), 36)

    def test_native_divisibility_error_is_preserved(self):
        with self.assertRaisesRegex(ValueError, 'not divisible'):
            self.trainer._prepare_rollout_batch(SimpleNamespace(dp_size=9), self.samples(35))

    def test_custom_encoder_and_collator_keep_the_driver_path(self):
        for name in ['encode_rollout_batch', '_collate_for_workers']:
            with self.subTest(method=name):
                original = getattr(self.trainer, name)
                group = SimpleNamespace(dp_size=9, prepare_rollout_batch=Mock(side_effect=AssertionError('remote')))
                with patch.object(self.trainer, name, wraps=original) as custom:
                    self.trainer._prepare_rollout_batch(group, self.samples())
                    custom.assert_called_once()
                    group.prepare_rollout_batch.assert_not_called()

    def test_missing_collector_fails_before_training(self):
        with self.assertRaisesRegex(RuntimeError, 'every DP collector'):
            self.trainer._prepare_rollout_batch(
                SimpleNamespace(dp_size=9, prepare_rollout_batch=lambda *a, **kw: {}), self.samples())

    def test_custom_encoder_replacement_samples_reach_collation_and_teacher(self):
        samples, replacement = self.samples(), self.samples()
        for sample in replacement:
            sample.encoded = self.encode(sample, self.template)
            sample.encoded.pop('_extra_kwargs')
            sample.encoded['input_ids'][0] += 100
        group = SimpleNamespace(dp_size=9, prepare_rollout_batch=Mock(side_effect=AssertionError('remote')))
        with patch.object(self.trainer, 'encode_rollout_batch', return_value=replacement):
            dispatch, _ = self.trainer._prepare_rollout_batch(group, samples)
        self.assertIs(samples[0], replacement[0])
        self.assertEqual(dispatch[0][0]['input_ids'][0, 0].item(), 100)
        group.prepare_rollout_batch.assert_not_called()

    def test_worker_preserves_preencoded_teacher_inputs(self):
        from swift.ray.megatron.data import resolve_batches

        sample = self.samples(1)[0]
        sample.encoded = {'input_ids': [91, 92, 93], 'media': [1], 'length': 3}
        result = self.worker().prepare_rollout_batch([copy.deepcopy(sample)], padding_to=4)
        self.assertEqual(self.calls, [])
        self.assertEqual(resolve_batches(result)[0]['input_ids'].tolist(), [[91, 92, 93]])
        self.assertIn('media', sample.encoded)

    def test_noncollector_does_not_encode_or_change_threads(self):
        self.assertIsNone(self.worker(False).prepare_rollout_batch(self.samples(), padding_to=4))
        self.assertEqual(self.calls, [])
        self.assertEqual(torch.get_num_threads(), 2)

    def test_worker_restores_threads_on_encode_collate_and_store_errors(self):
        for target in [
                'swift.rlhf_trainers.utils.encode_sample', 'swift.rlhf_trainers.utils.collate_to_grpo_micro_batch',
                'ray.put'
        ]:
            with self.subTest(target=target), patch(target, side_effect=RuntimeError('preparation failure')):
                with self.assertRaisesRegex(RuntimeError, 'preparation failure'):
                    self.worker().prepare_rollout_batch(self.samples(2), padding_to=4)
                self.assertEqual(torch.get_num_threads(), 2)

    def test_deterministic_encoder_preserves_rng_and_cuda_initialization(self):
        cpu = torch.get_rng_state()
        python, numpy = random.getstate(), np.random.get_state()
        cuda = torch.cuda.is_initialized()
        self.worker().prepare_rollout_batch(self.samples(2), padding_to=4)
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
        self.assertEqual(python, random.getstate())
        current = np.random.get_state()
        np.testing.assert_array_equal(numpy[1], current[1])
        self.assertEqual(numpy[0], current[0])
        self.assertEqual(numpy[2:], current[2:])
        self.assertEqual(cuda, torch.cuda.is_initialized())

    def test_global_normalization_uses_native_marker(self):
        for loss in ['grpo', 'dr_grpo', 'bnpo', 'cispo', 'dapo', 'fipo']:
            with self.subTest(loss=loss):
                self.args.loss_type = loss
                _, batches = self.trainer._prepare_rollout_batch(self.group(), self.samples())
                for batch in batches:
                    if loss in ['bnpo', 'cispo', 'dapo', 'fipo']:
                        self.assertEqual(batch.num_items_in_batch.item(), 72)
                    else:
                        self.assertIsNone(batch.num_items_in_batch)

    def test_latest_scores_and_advantages_are_resolved_each_time(self):
        from swift.ray.megatron.data import PreparedBatch, resolve_batches
        from swift.rl_core.data import GRPOBatch

        batch = GRPOBatch(torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), torch.tensor([3, 3]))
        ref = self.put({'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]])})
        payload = PreparedBatch(ref, batch)
        first = resolve_batches([payload])[0]
        batch.old_per_token_logps = torch.full((2, 3), 1.5)
        batch.ref_per_token_logps = torch.full((2, 3), 2.5)
        batch.advantages = torch.full((2, 3), 3.5)
        second = resolve_batches([payload])[0]
        self.assertIsNot(second, self.storage[ref])
        self.assertIsNot(second['grpo_batch'], batch)
        self.assertIsNone(first['grpo_batch'].advantages)
        for key in ['old_per_token_logps', 'ref_per_token_logps', 'advantages']:
            torch.testing.assert_close(getattr(second['grpo_batch'], key), getattr(batch, key))
        second.pop('input_ids')
        second['grpo_batch'].advantages = None
        self.assertIn('input_ids', self.storage[ref])
        self.assertIsNotNone(batch.advantages)

    def test_global_advantages_reach_every_prepared_microbatch(self):
        from swift.ray.megatron.data import resolve_batches
        from swift.rl_core.advantage import compute_advantages

        dispatch, batches = self.trainer._prepare_rollout_batch(self.group(), self.samples())
        trainer = self.trainer
        trainer.num_generations, trainer.reward_weights = 4, torch.tensor([1.0])
        trainer.kl_in_reward, trainer.beta, trainer._has_teacher = False, 0.0, False
        rewards = torch.arange(36, dtype=torch.float32).remainder(5)[:, None]
        for estimator, scale in [('grpo', 'group'), ('grpo', 'batch'), ('grpo', 'gdpo'), ('grpo', 'none'),
                                 ('rloo', 'none'), ('reinforce_plus_plus', 'batch')]:
            with self.subTest(estimator=estimator, scale=scale):
                trainer.advantage_estimator, trainer.scale_rewards = estimator, scale
                expected, _ = compute_advantages(rewards, trainer.reward_weights, 4, estimator, scale, False, 0.0)
                actual, _ = trainer.compute_advantages(rewards)
                trainer._scatter_advantages(batches, actual)
                received = [
                    inputs['grpo_batch'].advantages for rank in range(9) for inputs in resolve_batches(dispatch[rank])
                ]
                torch.testing.assert_close(torch.cat(received), expected[:, None].expand(36, 3), rtol=0, atol=0)

    def test_resolution_batches_ray_get_and_preserves_native_inputs(self):
        from swift.ray.megatron.data import PreparedBatch, resolve_batches
        from swift.rl_core.data import GRPOBatch

        batch = GRPOBatch(torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.tensor([1]))
        refs = [self.put({'input_ids': torch.tensor([i])}) for i in range(2)]
        native = {'prepared_inputs': 'native model key'}
        with patch('ray.get', side_effect=lambda values: [self.storage[ref] for ref in values]) as get:
            result = resolve_batches([PreparedBatch(refs[0], batch), native, PreparedBatch(refs[1], batch)])
            get.assert_called_once_with(refs)
            self.assertIs(result[1], native)
        with patch('ray.get') as get:
            self.assertIs(resolve_batches([native])[0], native)
            get.assert_not_called()
        with patch('ray.get', side_effect=RuntimeError('lost object')):
            with self.assertRaisesRegex(RuntimeError, 'lost object'):
                resolve_batches([PreparedBatch(refs[0], batch)])

    def test_worker_group_binds_native_preparation_signature(self):
        import inspect
        import ray

        from swift.ray.megatron.worker_group import WorkerGroup

        group = WorkerGroup('prepare', [])
        group.execute = Mock(return_value={})
        group._bind_decorated_methods(self.worker_cls)
        group.prepare_rollout_batch('shards', padding_to=4)
        group.execute.assert_called_once_with(
            'prepare_rollout_batch', 'shards', dispatch='dp', collect='dp', padding_to=4)
        actor = ray.remote(self.worker_cls)
        signature = inspect.signature(actor.__ray_metadata__.modified_class.prepare_rollout_batch)
        self.assertIn('padding_to', signature.parameters)


if __name__ == '__main__':
    unittest.main()
