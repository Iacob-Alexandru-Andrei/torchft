# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import MagicMock, create_autospec

import torch
from torch import nn, optim

from torchft.local_sgd import (
    _OptimizerStateFragment,
    _ParameterFragment,
    DesLoc,
)
from torchft.manager import Manager


def create_manager() -> MagicMock:
    """
    Creates a mock manager with some useful defaults for testing
    the optimizer's usage of the Manager
    """
    manager = create_autospec(Manager)
    manager._use_async_quorum = False
    manager.should_commit.return_value = True

    # Mock the allreduce function to average the tensors
    def mock_allreduce(tensor: torch.Tensor) -> torch.Tensor:
        # In a real scenario, this would be an allreduce operation.
        # For testing, we can just return the tensor as is,
        # or simulate averaging if needed.
        return tensor

    manager.allreduce.side_effect = mock_allreduce
    return manager


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DesLocTest(TestCase):
    def test_desloc_healthy_path(self) -> None:
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters())
        manager = create_manager()

        param_sync_every = 2
        optimizer_sync_every = [4]

        with DesLoc(
            manager, model, optimizer, param_sync_every, optimizer_sync_every
        ) as desloc:
            self.assertEqual(len(desloc._fragments), 1)  # Only param fragment initially
            self.assertFalse(desloc._is_opt_init)

            # --- Step 1 ---
            optimizer.step()
            self.assertTrue(desloc._is_opt_init)
            self.assertEqual(len(desloc._fragments), 3)  # Param + 2 optimizer states for AdamW

            param_fragment = desloc._fragments[0]
            opt_frag_1 = desloc._fragments[1]
            opt_frag_2 = desloc._fragments[2]

            self.assertEqual(param_fragment._local_step, 1)
            self.assertEqual(opt_frag_1._local_step, 1)
            self.assertEqual(opt_frag_2._local_step, 1)
            self.assertEqual(manager.start_quorum.call_count, 0)

            # --- Step 2 (Param Sync) ---
            optimizer.step()
            self.assertEqual(param_fragment._local_step, 0)  # Reset after sync
            self.assertEqual(opt_frag_1._local_step, 2)
            self.assertEqual(opt_frag_2._local_step, 2)
            self.assertEqual(manager.start_quorum.call_count, 1)
            self.assertEqual(manager.should_commit.call_count, 1)
            # 4 params in SimpleModel, so 4 allreduce calls
            self.assertEqual(manager.allreduce.call_count, 4)

            # --- Step 3 ---
            optimizer.step()
            self.assertEqual(param_fragment._local_step, 1)
            self.assertEqual(opt_frag_1._local_step, 3)
            self.assertEqual(opt_frag_2._local_step, 3)
            self.assertEqual(manager.start_quorum.call_count, 1)

            # --- Step 4 (Param + Optimizer Sync) ---
            manager.allreduce.reset_mock()
            optimizer.step()
            self.assertEqual(param_fragment._local_step, 0)
            self.assertEqual(opt_frag_1._local_step, 0)
            self.assertEqual(opt_frag_2._local_step, 0)
            self.assertEqual(manager.start_quorum.call_count, 2)
            self.assertEqual(manager.should_commit.call_count, 2)

            # 4 params + 4 exp_avg + 4 exp_avg_sq = 12 allreduce calls
            self.assertEqual(manager.allreduce.call_count, 12)

    def test_desloc_recovery_path(self) -> None:
        model = SimpleModel()
        optimizer = optim.AdamW(model.parameters())
        manager = create_manager()

        param_sync_every = 2
        optimizer_sync_every = [2]

        with DesLoc(
            manager, model, optimizer, param_sync_every, optimizer_sync_every
        ) as desloc:
            # --- Step 1: Run one step to populate optimizer state ---
            optimizer.step()

            # --- Save current state ---
            model_state_before_sync = {
                name: p.clone() for name, p in model.named_parameters()
            }
            optimizer_state_before_sync = {
                id(p): {k: v.clone() for k, v in s.items() if torch.is_tensor(v)}
                for p, s in optimizer.state.items()
            }

            # --- Step 2: Trigger sync with failure ---
            manager.should_commit.return_value = False
            optimizer.step()

            # --- Verify state restoration ---
            self.assertEqual(manager.start_quorum.call_count, 1)
            self.assertEqual(manager.should_commit.call_count, 1)

            # Check model parameters
            for name, p in model.named_parameters():
                self.assertTrue(torch.equal(p, model_state_before_sync[name]))

            # Check optimizer state
            for p, s in optimizer.state.items():
                for k, v in s.items():
                    if torch.is_tensor(v):
                        self.assertTrue(
                            torch.equal(v, optimizer_state_before_sync[id(p)][k])
                        )

            # --- Check local step counters ---
            # They should not be reset because the sync failed
            for fragment in desloc._fragments:
                self.assertEqual(fragment._local_step, 2)

    def test_desloc_lazy_init(self) -> None:
        # --- Test with SGD (no state) ---
        model_sgd = SimpleModel()
        optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
        manager_sgd = create_manager()

        with DesLoc(
            manager_sgd, model_sgd, optimizer_sgd, param_sync_every=2, optimizer_sync_every=[]
        ) as desloc_sgd:
            self.assertFalse(desloc_sgd._is_opt_init)
            self.assertEqual(len(desloc_sgd._fragments), 1)

            optimizer_sgd.step()

            self.assertTrue(desloc_sgd._is_opt_init)
            # No new fragments should be added for SGD as it has no tensor state
            self.assertEqual(len(desloc_sgd._fragments), 1)

        # --- Test with AdamW (with state) ---
        model_adam = SimpleModel()
        optimizer_adam = optim.AdamW(model_adam.parameters())
        manager_adam = create_manager()

        with DesLoc(
            manager_adam,
            model_adam,
            optimizer_adam,
            param_sync_every=2,
            optimizer_sync_every=[4, 4],
        ) as desloc_adam:
            self.assertFalse(desloc_adam._is_opt_init)
            self.assertEqual(len(desloc_adam._fragments), 1)

            optimizer_adam.step()

            self.assertTrue(desloc_adam._is_opt_init)
            # AdamW has 'exp_avg' and 'exp_avg_sq' states
            self.assertEqual(len(desloc_adam._fragments), 3)
            self.assertIsInstance(desloc_adam._fragments[0], _ParameterFragment)
            self.assertIsInstance(desloc_adam._fragments[1], _OptimizerStateFragment)
            self.assertIsInstance(desloc_adam._fragments[2], _OptimizerStateFragment)
            self.assertEqual(desloc_adam._fragments[1].state_key, "exp_avg")
            self.assertEqual(desloc_adam._fragments[2].state_key, "exp_avg_sq")

    def test_desloc_sync_correctness(self) -> None:
        # --- Setup two "workers" ---
        model1 = SimpleModel()
        optimizer1 = optim.AdamW(model1.parameters())
        manager1 = create_manager()

        model2 = SimpleModel()
        optimizer2 = optim.AdamW(model2.parameters())
        manager2 = create_manager()

        # --- Set different initial weights for workers ---
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p1.data.fill_(1.0)
                p2.data.fill_(3.0)

        param_sync_every = 1
        optimizer_sync_every = [1]

        with DesLoc(
            manager1, model1, optimizer1, param_sync_every, optimizer_sync_every
        ) as desloc1, DesLoc(
            manager2, model2, optimizer2, param_sync_every, optimizer_sync_every
        ) as desloc2:

            # --- Manually simulate allreduce ---
            def simulated_allreduce(deslocs):
                all_fragments = [d._fragments for d in deslocs]
                for frags in zip(*all_fragments):
                    # Simulate parameter sync
                    if isinstance(frags[0], _ParameterFragment):
                        params = [list(f._model.parameters()) for f in frags]
                        for ps in zip(*params):
                            avg_p = torch.stack([p.data for p in ps]).mean(dim=0)
                            for p in ps:
                                p.data.copy_(avg_p)
                    # Simulate optimizer state sync
                    elif isinstance(frags[0], _OptimizerStateFragment):
                        key = frags[0].state_key
                        states = [
                            [
                                f._optimizer.state[p][key]
                                for p in f._model.parameters()
                                if p in f._optimizer.state
                                and key in f._optimizer.state[p]
                            ]
                            for f in frags
                        ]
                        for ss in zip(*states):
                            avg_s = torch.stack(ss).mean(dim=0)
                            for s in ss:
                                s.copy_(avg_s)

            # --- Run step to trigger sync ---
            optimizer1.step()
            optimizer2.step()
            simulated_allreduce([desloc1, desloc2])

            # --- Verify that parameters are averaged ---
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                self.assertTrue(torch.all(torch.eq(p1, p2)))
                self.assertTrue(torch.all(torch.eq(p1, torch.full_like(p1, 2.0))))

            # --- Verify that optimizer states are averaged ---
            # AdamW initializes state lazily, so we need to check after a step
            optimizer1.step()
            optimizer2.step()
            simulated_allreduce([desloc1, desloc2])

            for (p1, s1), (p2, s2) in zip(
                optimizer1.state.items(), optimizer2.state.items()
            ):
                for key in s1:
                    if torch.is_tensor(s1[key]):
                        self.assertTrue(torch.equal(s1[key], s2[key]))
