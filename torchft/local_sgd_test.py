# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Set
from unittest import TestCase
from unittest.mock import create_autospec, MagicMock

import torch
from parameterized import parameterized
from torch import nn, optim, Tensor
from torch.distributed.distributed_c10d import Work
from torch.distributed.tensor import DTensor
from torchft.local_sgd import DESLoc, DiLoCo, extract_local_tensor, LocalSGD
from torchft.manager import Manager
from torchft.work import _DummyWork


def create_manager() -> MagicMock:
    """
    Creates a mock manager with some useful defaults for testing
    the optimizer's usage of the Manager
    """
    manager = create_autospec(Manager)

    manager.errored.return_value = None

    def mock_allreduce(tensor: torch.Tensor, should_quantize: bool = False) -> Work:
        return _DummyWork(tensor)

    manager.allreduce.side_effect = mock_allreduce

    return manager


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _params_dict(m: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.data for name, p in m.named_parameters()}


def _copy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.clone().detach() for name, value in state_dict.items()}


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.w2 = nn.Parameter(torch.tensor([3.0, 4.0, 5.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w1.unsqueeze(0).T + self.w2.sum()


class LocalSGDTest(TestCase):
    def test_local_sgd_healthy(self) -> None:
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters())
        manager = create_manager()
        with LocalSGD(manager, model, optimizer, sync_every=2) as local_sgd:
            self.assertEqual(local_sgd._local_step, 0)
            inp = torch.rand(2, 3)
            loss = model(inp).mean()
            loss.backward()
            optimizer.step()

            self.assertEqual(local_sgd._local_step, 1)
            self.assertEqual(manager.start_quorum.call_count, 0)
            loss = model(inp).mean()
            loss.backward()
            optimizer.step()
            self.assertEqual(manager.start_quorum.call_count, 1)

            manager.should_commit.return_value = True
            self.assertEqual(local_sgd._local_step, 0)
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertEqual(manager.allreduce.call_count, 4)

    def test_extract_local_tensor(self) -> None:
        regular_tensor = torch.rand(3, 3)
        regular_result = extract_local_tensor(regular_tensor)

        self.assertTrue(torch.equal(regular_result, regular_tensor))
        self.assertIsNone(regular_result.grad)
        self.assertNotEqual(id(regular_result), id(regular_tensor))
        local_tensor = torch.rand(3, 3)
        dtensor = MagicMock(spec=DTensor)
        dtensor.to_local.return_value = local_tensor
        dtensor_result = extract_local_tensor(dtensor)

        self.assertTrue(torch.equal(dtensor_result, local_tensor))
        self.assertIsNone(dtensor_result.grad)
        self.assertNotEqual(id(dtensor_result), id(local_tensor))
        dtensor.to_local.assert_called_once()

    def test_local_sgd_recovery(self) -> None:
        model = SimpleModel()
        optimizer = optim.SGD(model.parameters())
        manager = create_autospec(Manager)

        with LocalSGD(manager, model, optimizer, sync_every=2) as local_sgd:
            og_state_dict = _copy_state_dict(model.state_dict())

            inp = torch.rand(2, 3)

            loss = model(inp).mean()
            loss.backward()
            optimizer.step()

            # Check that the model's state dict has been updated
            for name, param in model.state_dict().items():
                # Ensure the parameter has changed
                self.assertFalse(
                    torch.equal(og_state_dict[name], param),
                    f"Parameter {name} did not change.",
                )
            self.assertEqual(local_sgd._local_step, 1)


class DiLoCoTest(TestCase):
    def test_diloco_healthy(self) -> None:
        model = SimpleModel()

        # Setup optimizers
        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )

        manager = create_manager()
        manager._use_async_quorum = False
        with DiLoCo(
            manager, [model], inner_optimizer, outer_optimizer, sync_every=2
        ) as diloco:
            parameter_count = len(list(model.parameters()))
            initial_outer_opt_state = outer_optimizer.state_dict()
            self.assertEqual(initial_outer_opt_state["state"], {})

            self.assertEqual(diloco._local_step, 0)
            torch.testing.assert_close(
                diloco._fragments[0].original_parameters, _params_dict(model)
            )
            inp = torch.rand(2, 3)
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(diloco._local_step, 1)
            manager.current_step.return_value = 0
            manager.should_commit.return_value = True
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            self.assertEqual(diloco._local_step, 0)
            self.assertEqual(manager.start_quorum.call_count, 1)
            torch.testing.assert_close(
                diloco._fragments[0].original_parameters, _params_dict(model)
            )
            self.assertEqual(manager.should_commit.call_count, 1)
            self.assertEqual(manager.allreduce.call_count, parameter_count)

            outer_opt_state = outer_optimizer.state_dict()
            self.assertEqual(len(outer_opt_state["state"]), parameter_count)

    @parameterized.expand(
        [
            ("bucketized_should_use_fewer_calls", True, True),
            ("non_bucketized_should_call_per_param", False, False),
        ]
    )
    def test_diloco_allreduce_call_efficiency(
        self,
        name: str,
        use_bucketization: bool,
        expect_fewer_calls: bool,
    ) -> None:
        model = SimpleModel()

        inner_optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer = torch.optim.SGD(
            model.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )

        manager = create_manager()
        manager._use_async_quorum = False
        manager.should_commit.return_value = True

        with DiLoCo(
            manager,
            [model],
            inner_optimizer,
            outer_optimizer,
            sync_every=2,
            use_bucketization=use_bucketization,
        ) as diloco:
            inp = torch.rand(2, 3)
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            manager.current_step.return_value = 0
            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            loss = model(inp).mean()
            loss.backward()
            inner_optimizer.step()

            allreduce_calls = manager.allreduce.call_count
            param_count = len([p for p in model.parameters() if p.requires_grad])

            if expect_fewer_calls:
                self.assertLess(int(allreduce_calls), int(param_count))
            else:
                self.assertEqual(int(allreduce_calls), int(param_count))

    def test_bucketization_correctness(self) -> None:
        model = TinyModel()
        inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        outer_opt = torch.optim.SGD(model.parameters(), lr=0.1)

        manager = create_autospec(Manager)
        manager._use_async_quorum = False
        manager.should_commit.return_value = True

        # Define fake allreduce: multiplies buffer by 2
        def fake_allreduce(tensor: Tensor, should_quantize: bool) -> Work:
            tensor.mul_(2)
            return _DummyWork(tensor)

        manager.allreduce.side_effect = fake_allreduce

        diloco = DiLoCo(
            manager, [model], inner_opt, outer_opt, sync_every=2, use_bucketization=True
        )
        diloco._fragments[0].bucket_cap_mb = 10 * 1024 * 1024

        # Manually assign fake gradients
        grads = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
        for g, (name, param) in zip(grads, model.named_parameters()):
            diloco._fragments[0]._grads[name] = g.clone()

        # Run only bucketized logic
        diloco._fragments[0]._average_grads()

        # The parameter gradients should not be set
        for param in model.parameters():
            self.assertEqual(param.grad, None)

        diloco._fragments[0]._set_grads()

        # Expect grads to have been doubled
        expected_grads = [g * 2 for g in grads]
        for param, expected in zip(model.parameters(), expected_grads):
            torch.testing.assert_close(param.grad, expected, rtol=1e-5, atol=1e-8)

    def test_gradient_correctness(self) -> None:
        model = TinyModel()
        inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        outer_opt = torch.optim.SGD(model.parameters(), lr=0.1)

        manager = create_autospec(Manager)
        manager._use_async_quorum = False
        manager.should_commit.return_value = True

        # Define fake allreduce: multiplies buffer by 2
        def fake_allreduce(tensor: Tensor, should_quantize: bool) -> Work:
            tensor.mul_(2)
            return _DummyWork(tensor)

        manager.allreduce.side_effect = fake_allreduce

        diloco = DiLoCo(manager, [model], inner_opt, outer_opt, sync_every=2)

        # save original parameters
        diloco._fragments[0].save_parameters()

        # change the model's parameters
        for p in model.parameters():
            p.data.add_(2)

        # calculate and set the gradients
        diloco._fragments[0]._save_grads()

        # calculate
        diloco._fragments[0]._average_grads()

        # The parameter gradients should not be set
        for param in model.parameters():
            self.assertEqual(param.grad, None)

        diloco._fragments[0]._set_grads()

        # we added 2 to the parameters, then multiplied the gradients by 2
        # so we should expect the model's gradient to be -4
        expected_grad = -4
        for param in model.parameters():
            assert param.grad is not None
            t = torch.empty_like(param.grad)
            t.fill_(expected_grad)
            torch.testing.assert_close(param.grad, t)


class DESLocTest(TestCase):
    def _train_step(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        inp = torch.rand(2, 3)
        loss = model(inp).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _float_state_keys(self, optimizer: optim.Optimizer) -> Set[str]:
        keys: Set[str] = set()
        for state in optimizer.state.values():
            for key, value in state.items():
                if (
                    isinstance(value, torch.Tensor)
                    and torch.is_floating_point(value)
                    and value.numel() > 1
                ):
                    keys.add(str(key))
        return keys

    def test_desloc_discovers_state_keys_dynamically(self) -> None:
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        manager = create_manager()

        with DESLoc(manager, model, optimizer, sync_every=1000) as desloc:
            self._train_step(model, optimizer)
            expected_state_keys = self._float_state_keys(optimizer)
            self.assertGreater(len(expected_state_keys), 0)
            self.assertEqual(
                set(desloc._optimizer_state_sync_every.keys()),
                expected_state_keys,
            )

    def test_desloc_per_key_frequency_sync(self) -> None:
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        manager = create_manager()
        manager.should_commit.return_value = True

        with DESLoc(
            manager,
            model,
            optimizer,
            sync_every=1000,
            optimizer_sync_every={"exp_avg": 2, "exp_avg_sq": 4},
        ):
            for _ in range(4):
                self._train_step(model, optimizer)

            state_param_count = len(optimizer.state)
            self.assertEqual(manager.start_quorum.call_count, 2)
            self.assertEqual(manager.allreduce.call_count, state_param_count * 3)

    def test_desloc_dict_missing_key_falls_back_to_sync_every(self) -> None:
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        manager = create_manager()

        with DESLoc(
            manager,
            model,
            optimizer,
            sync_every=7,
            optimizer_sync_every={"exp_avg": 2},
        ) as desloc:
            self._train_step(model, optimizer)
            self.assertEqual(desloc._optimizer_state_sync_every["exp_avg"], 2)
            self.assertEqual(desloc._optimizer_state_sync_every["exp_avg_sq"], 7)

    def test_desloc_outer_optimizer_commit_false_rolls_back_params(self) -> None:
        model = TinyModel()
        inner_optimizer = optim.SGD(model.parameters(), lr=1.0)
        outer_optimizer = optim.SGD(model.parameters(), lr=0.5)
        manager = create_manager()
        manager.should_commit.return_value = False

        with DESLoc(
            manager,
            model,
            inner_optimizer,
            sync_every=1,
            outer_optimizer=outer_optimizer,
        ):
            initial_state = _copy_state_dict(model.state_dict())
            for param in model.parameters():
                param.grad = torch.ones_like(param)
            inner_optimizer.step()

            for name, param in model.state_dict().items():
                torch.testing.assert_close(param, initial_state[name])

    def test_desloc_outer_optimizer_commit_true_updates_reference(self) -> None:
        model = TinyModel()
        inner_optimizer = optim.SGD(model.parameters(), lr=1.0)
        outer_optimizer = optim.SGD(model.parameters(), lr=0.5)
        manager = create_manager()
        manager.should_commit.return_value = True

        with DESLoc(
            manager,
            model,
            inner_optimizer,
            sync_every=1,
            outer_optimizer=outer_optimizer,
        ) as desloc:
            initial_state = _copy_state_dict(model.state_dict())
            for param in model.parameters():
                param.grad = torch.ones_like(param)
            inner_optimizer.step()

            expected_state = {
                name: value - 0.5 for name, value in initial_state.items()
            }
            for name, param in model.state_dict().items():
                torch.testing.assert_close(param, expected_state[name])

            for name, reference in desloc._reference_parameters.items():
                torch.testing.assert_close(reference, expected_state[name])

    def test_desloc_float_tensor_filter(self) -> None:
        model = TinyModel()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        manager = create_manager()

        for param in model.parameters():
            optimizer.state[param]["float_vector"] = torch.ones_like(param)
            optimizer.state[param]["float_scalar"] = torch.tensor(1.0)
            optimizer.state[param]["int_vector"] = torch.ones_like(
                param, dtype=torch.int64
            )

        with DESLoc(
            manager,
            model,
            optimizer,
            sync_every=1000,
            optimizer_sync_every=2,
        ) as desloc:
            for param in model.parameters():
                param.grad = torch.ones_like(param)
            optimizer.step()

            self.assertIn("float_vector", desloc._optimizer_state_sync_every)
            self.assertNotIn("float_scalar", desloc._optimizer_state_sync_every)
            self.assertNotIn("int_vector", desloc._optimizer_state_sync_every)
