# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os
import re
import sys
import threading
import traceback
from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import field
from datetime import timedelta
from typing import Any, cast, Dict, List, Optional
from unittest import skipIf, TestCase

import torch
from parameterized import parameterized
from torch import nn, optim
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.distributed.tensor import DTensor, Replicate
from torchft._test.diloco_trainer import DiLoCoTrainer, MultiMyModel
from torchft._torchft import LighthouseServer
from torchft.local_sgd import DESLoc, DiLoCo, LocalSGD
from torchft.manager import Manager
from torchft.manager_integ_test import (
    EventInjector,
    EventInjectorEvent,
    MyModel,
    Runner,
)
from torchft.process_group import (
    FakeProcessGroupWrapper,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
)

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _ensure_fr_reset_api_for_test() -> None:
    # Some torch builds used in local CPU test environments don't expose this
    # Flight Recorder reset API, but Manager currently assumes it exists.
    c10d = torch._C._distributed_c10d
    if hasattr(c10d, "_reset_fr_recording_nccl"):
        return
    setattr(c10d, "_reset_fr_recording_nccl", lambda: None)


def _ensure_cpu_accelerator_api_for_test() -> None:
    # Some CPU-only torch builds report accelerator availability as true,
    # which can trigger aborts in torch.accelerator.synchronize().
    if torch.cuda.is_available():
        return
    if not torch.accelerator.is_available():
        return
    setattr(torch.accelerator, "is_available", lambda: False)
    setattr(torch.accelerator, "synchronize", lambda: None)


class _DESLocScalarModel(nn.Module):
    def __init__(self, init_weight: float = 10.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([init_weight, init_weight], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _DESLocRecordingOptimizer(optim.Optimizer):
    def __init__(self, params, lr: float = 1.0) -> None:
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.last_local_after_step: Dict[int, torch.Tensor] = {}
        self.last_state_after_step: Dict[int, Dict[str, torch.Tensor]] = {}

    def step(self, closure=None):  # pyre-ignore[2]
        del closure
        self.last_local_after_step = {}
        self.last_state_after_step = {}
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                state_a = cast(Optional[torch.Tensor], state.get("state_a"))
                state_b = cast(Optional[torch.Tensor], state.get("state_b"))
                if state_a is None:
                    state_a = torch.zeros_like(p)
                if state_b is None:
                    state_b = torch.zeros_like(p)

                grad = p.grad.detach()
                state_a = state_a + grad
                state_b = state_b + grad * grad
                state["state_a"] = state_a
                state["state_b"] = state_b

                p.data.add_(grad, alpha=-lr)
                p_id = id(p)
                self.last_local_after_step[p_id] = p.data.detach().clone()
                self.last_state_after_step[p_id] = {
                    "state_a": state_a.detach().clone(),
                    "state_b": state_b.detach().clone(),
                }

        return None


def desloc_train_loop(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
    train_loop_args: dict[str, Any] = {},
) -> Dict[str, object]:
    _ensure_fr_reset_api_for_test()
    _ensure_cpu_accelerator_api_for_test()
    sync_every = int(train_loop_args.get("sync_every", 2))
    optimizer_sync_every = train_loop_args.get("optimizer_sync_every", None)
    max_local_steps = int(train_loop_args.get("max_local_steps", 4))
    replica_grads = cast(dict[int, float], train_loop_args.get("replica_grads", {}))
    grad_value = float(replica_grads.get(runner.replica_id, 1.0))
    use_outer_optimizer = bool(train_loop_args.get("use_outer_optimizer", False))
    outer_lr = float(train_loop_args.get("outer_lr", 1.0))

    with ExitStack() as stack:

        def load_state_dict(state_dict: Dict[str, Dict[str, object]]) -> None:
            m.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optim"])
            if outer_optimizer is not None and "outer_optim" in state_dict:
                outer_optimizer.load_state_dict(state_dict["outer_optim"])

        def state_dict() -> Dict[str, Dict[str, object]]:
            payload: Dict[str, Dict[str, object]] = {
                "model": m.state_dict(),
                "optim": optimizer.state_dict(),
            }
            if outer_optimizer is not None:
                payload["outer_optim"] = outer_optimizer.state_dict()
            return payload

        if device.type == "cuda":
            pg = ProcessGroupBabyNCCL()
        else:
            pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=2,
            load_state_dict=load_state_dict,
            state_dict=state_dict,
            replica_id=str(runner.replica_id),
            store_addr="localhost",
            store_port=store_port,
            rank=rank,
            world_size=runner.world_size,
            lighthouse_addr=runner.lighthouse_address,
            port=19530 + runner.replica_id,
            timeout=timedelta(seconds=10),
            quorum_timeout=timedelta(seconds=10),
            # pyre-fixme[6]: Incompatible parameter type
            **runner.manager_args,
        )
        stack.callback(lambda: manager.shutdown(wait=False))

        m = _DESLocScalarModel().to(device)
        optimizer = _DESLocRecordingOptimizer(m.parameters(), lr=1.0)
        outer_optimizer = (
            optim.SGD(m.parameters(), lr=outer_lr) if use_outer_optimizer else None
        )

        records: List[Dict[str, float]] = []
        with DESLoc(
            manager,
            m,
            optimizer,
            sync_every=sync_every,
            optimizer_sync_every=optimizer_sync_every,
            outer_optimizer=outer_optimizer,
            backup_device=device,
        ):
            for local_step in range(1, max_local_steps + 1):
                m.weight.grad = torch.full_like(m.weight, grad_value)
                optimizer.step()
                optimizer.zero_grad()

                p_id = id(m.weight)
                pre_sync_local_param = optimizer.last_local_after_step[p_id]
                pre_sync_local_states = optimizer.last_state_after_step[p_id]
                post_state_a = cast(torch.Tensor, optimizer.state[m.weight]["state_a"])
                post_state_b = cast(torch.Tensor, optimizer.state[m.weight]["state_b"])

                records.append(
                    {
                        "local_step": float(local_step),
                        "pre_sync_local_param": float(
                            pre_sync_local_param.detach().cpu()[0].item()
                        ),
                        "post_param": float(m.weight.detach().cpu()[0].item()),
                        "pre_sync_local_state_a": float(
                            pre_sync_local_states["state_a"].detach().cpu()[0].item()
                        ),
                        "post_state_a": float(post_state_a.detach().cpu()[0].item()),
                        "pre_sync_local_state_b": float(
                            pre_sync_local_states["state_b"].detach().cpu()[0].item()
                        ),
                        "post_state_b": float(post_state_b.detach().cpu()[0].item()),
                        "manager_step": float(manager.current_step()),
                    }
                )

                if manager.current_step() >= 3:
                    break

                runner.event_injector.check(rank, manager.current_step())

        return {"replica_id": runner.replica_id, "records": records}
    return {}


def local_sgd_train_loop(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
    train_loop_args: dict[str, Any] = {},
) -> Dict[str, Dict[str, object]]:
    with ExitStack() as stack:

        def load_state_dict(state_dict: Dict[str, Dict[str, object]]) -> None:
            m.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optim"])

        def state_dict() -> Dict[str, Dict[str, object]]:
            return {
                "model": m.state_dict(),
                "optim": optimizer.state_dict(),
            }

        print(f"worker {runner.replica_id=} {rank=} {runner.world_size=} starting")

        if device.type == "cuda":
            pg = ProcessGroupBabyNCCL()
        else:
            pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=2,
            load_state_dict=load_state_dict,
            state_dict=state_dict,
            replica_id=str(runner.replica_id),
            store_addr="localhost",
            store_port=store_port,
            rank=rank,
            world_size=runner.world_size,
            lighthouse_addr=runner.lighthouse_address,
            port=19530 + runner.replica_id,
            timeout=timedelta(seconds=10),
            # pyre-fixme[6]: Incompatible parameter type
            **runner.manager_args,
        )
        stack.callback(lambda: manager.shutdown(wait=False))

        m: nn.Module = MyModel().to(device)

        optimizer: optim.Optimizer = optim.Adam(m.parameters())
        criterion = nn.CrossEntropyLoss()

        with LocalSGD(manager, m, optimizer, sync_every=2) as local_sgd:
            while True:
                inputs = torch.rand(2, 3).to(device)
                labels = torch.randint(4, (2,)).to(device)

                optimizer.zero_grad()
                out = m(inputs)
                loss = criterion(out, labels)
                loss.backward()

                optimizer.step()

                if manager.current_step() >= 4:
                    break

                runner.event_injector.check(rank, manager.current_step())

        # return state_dict so we can check consistency
        return state_dict()
    return {}


def diloco_train_loop(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
    train_loop_args: dict[str, Any] = {},
) -> Dict[str, Dict[str, object]]:
    model_state_dict = train_loop_args.get("model_state_dict", {})
    n_fragments = train_loop_args.get("n_fragments", 1)
    diloco_args = train_loop_args.get("diloco_args", {})

    with ExitStack() as stack:
        trainer = DiLoCoTrainer(
            rank, store_port, device, runner, model_state_dict, n_fragments, diloco_args
        )
        stack.callback(trainer.manager.shutdown)
        return trainer.train_loop()
    return {}


def assert_equal_global_state(
    n_fragments: int,
    rep0: dict[str, dict[str, dict[str, dict[str, object]]]],
    rep1: dict[str, dict[str, dict[str, dict[str, object]]]],
) -> None:
    """
    Asserts that the global state of the two replicas are equal
    """
    for step in rep0.keys():
        for i in range(n_fragments):
            torch.testing.assert_close(
                rep1[step]["user"][f"StreamingDiLoCoFragment_{i}"][
                    "original_parameters"
                ],
                rep0[step]["user"][f"StreamingDiLoCoFragment_{i}"][
                    "original_parameters"
                ],
                check_device=False,
                msg=f"{step=} {i=}",
            )
            # Check all outer optimizers
            torch.testing.assert_close(
                cast(
                    dict[str, dict[str, torch.Tensor]],
                    rep1[step]["user"][f"StreamingDiLoCoFragment_{i}"][
                        "outer_optimizer"
                    ],
                ),
                cast(
                    dict[str, dict[str, torch.Tensor]],
                    rep0[step]["user"][f"StreamingDiLoCoFragment_{i}"][
                        "outer_optimizer"
                    ],
                ),
                check_device=False,
            )


class LocalSGDIntegTest(TestCase):
    def _run_desloc_replicas(self, train_loop_args: dict[str, Any]) -> Dict[int, object]:
        lighthouse = LighthouseServer(bind="[::]:0", min_replicas=2)
        num_replicas = 2
        futures = []
        results: Dict[int, object] = {}

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id in range(num_replicas):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=EventInjector(),
                    train_loop=desloc_train_loop,
                    use_cuda=False,
                    manager_args={
                        "use_async_quorum": False,
                        # Keep deterministic startup to avoid checkpoint-heal noise
                        # in pre/post averaging assertions.
                        "init_sync": False,
                    },
                    train_loop_args=train_loop_args,
                )
                futures.append(executor.submit(runner.run_replica))

            for fut in as_completed(futures):
                payload = cast(dict[str, object], fut.result()[0])
                replica_id = cast(int, payload["replica_id"])
                results[replica_id] = payload

        lighthouse.shutdown()
        return results

    def _get_step_record(
        self, payload: dict[str, object], local_step: int
    ) -> dict[str, float]:
        records = cast(list[dict[str, float]], payload["records"])
        for record in records:
            if int(record["local_step"]) == local_step:
                return record
        raise AssertionError(f"Missing record for local_step={local_step}")

    def test_desloc_param_averaging_pre_post(self) -> None:
        results = self._run_desloc_replicas(
            {
                "sync_every": 2,
                "optimizer_sync_every": 1000,
                "max_local_steps": 4,
                "replica_grads": {
                    0: 1.0,
                    1: 3.0,
                },
            }
        )

        rep0 = cast(dict[str, object], results[0])
        rep1 = cast(dict[str, object], results[1])

        for local_step in [2, 4]:
            step0 = self._get_step_record(rep0, local_step)
            step1 = self._get_step_record(rep1, local_step)
            expected_avg = (
                step0["pre_sync_local_param"] + step1["pre_sync_local_param"]
            ) / 2.0
            self.assertAlmostEqual(step0["post_param"], expected_avg, places=6)
            self.assertAlmostEqual(step1["post_param"], expected_avg, places=6)

        for local_step in [1, 3]:
            step0 = self._get_step_record(rep0, local_step)
            step1 = self._get_step_record(rep1, local_step)
            self.assertAlmostEqual(
                step0["post_param"], step0["pre_sync_local_param"], places=6
            )
            self.assertAlmostEqual(
                step1["post_param"], step1["pre_sync_local_param"], places=6
            )

    def test_desloc_outer_optimizer_averaging_pre_post(self) -> None:
        results = self._run_desloc_replicas(
            {
                "sync_every": 2,
                "optimizer_sync_every": 1000,
                "max_local_steps": 4,
                "use_outer_optimizer": True,
                "outer_lr": 1.0,
                "replica_grads": {
                    0: 1.0,
                    1: 3.0,
                },
            }
        )

        rep0 = cast(dict[str, object], results[0])
        rep1 = cast(dict[str, object], results[1])

        for local_step in [2, 4]:
            step0 = self._get_step_record(rep0, local_step)
            step1 = self._get_step_record(rep1, local_step)
            expected_avg = (
                step0["pre_sync_local_param"] + step1["pre_sync_local_param"]
            ) / 2.0
            self.assertAlmostEqual(step0["post_param"], expected_avg, places=6)
            self.assertAlmostEqual(step1["post_param"], expected_avg, places=6)

        for local_step in [1, 3]:
            step0 = self._get_step_record(rep0, local_step)
            step1 = self._get_step_record(rep1, local_step)
            self.assertAlmostEqual(
                step0["post_param"], step0["pre_sync_local_param"], places=6
            )
            self.assertAlmostEqual(
                step1["post_param"], step1["pre_sync_local_param"], places=6
            )

    def test_desloc_optimizer_state_averaging_pre_post(self) -> None:
        results = self._run_desloc_replicas(
            {
                "sync_every": 1000,
                "optimizer_sync_every": {
                    "state_a": 2,
                    "state_b": 4,
                },
                "max_local_steps": 4,
                "replica_grads": {
                    0: 1.0,
                    1: 3.0,
                },
            }
        )

        rep0 = cast(dict[str, object], results[0])
        rep1 = cast(dict[str, object], results[1])

        step2_rep0 = self._get_step_record(rep0, 2)
        step2_rep1 = self._get_step_record(rep1, 2)
        expected_state_a_step2 = (
            step2_rep0["pre_sync_local_state_a"] + step2_rep1["pre_sync_local_state_a"]
        ) / 2.0
        self.assertAlmostEqual(step2_rep0["post_state_a"], expected_state_a_step2, places=6)
        self.assertAlmostEqual(step2_rep1["post_state_a"], expected_state_a_step2, places=6)
        self.assertAlmostEqual(
            step2_rep0["post_state_b"], step2_rep0["pre_sync_local_state_b"], places=6
        )
        self.assertAlmostEqual(
            step2_rep1["post_state_b"], step2_rep1["pre_sync_local_state_b"], places=6
        )

        for local_step in [1, 3]:
            step0 = self._get_step_record(rep0, local_step)
            step1 = self._get_step_record(rep1, local_step)
            self.assertAlmostEqual(
                step0["post_state_a"], step0["pre_sync_local_state_a"], places=6
            )
            self.assertAlmostEqual(
                step1["post_state_a"], step1["pre_sync_local_state_a"], places=6
            )
            self.assertAlmostEqual(
                step0["post_state_b"], step0["pre_sync_local_state_b"], places=6
            )
            self.assertAlmostEqual(
                step1["post_state_b"], step1["pre_sync_local_state_b"], places=6
            )

        step4_rep0 = self._get_step_record(rep0, 4)
        step4_rep1 = self._get_step_record(rep1, 4)
        expected_state_a_step4 = (
            step4_rep0["pre_sync_local_state_a"] + step4_rep1["pre_sync_local_state_a"]
        ) / 2.0
        expected_state_b_step4 = (
            step4_rep0["pre_sync_local_state_b"] + step4_rep1["pre_sync_local_state_b"]
        ) / 2.0
        self.assertAlmostEqual(step4_rep0["post_state_a"], expected_state_a_step4, places=6)
        self.assertAlmostEqual(step4_rep1["post_state_a"], expected_state_a_step4, places=6)
        self.assertAlmostEqual(step4_rep0["post_state_b"], expected_state_b_step4, places=6)
        self.assertAlmostEqual(step4_rep1["post_state_b"], expected_state_b_step4, places=6)

    # TODO: race condition due to using NCCL in threads causes manager allreduce to sometimes not be correct
    # Because of that the test is disabled for cuda
    @parameterized.expand(
        [
            # (True,),
            (False,),
        ]
    )
    def test_local_sgd_recovery(self, use_cuda: bool) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")
        if sys.platform == "darwin":
            self.skipTest("not reliable on mac")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        event_injectors = [
            EventInjector(),
            EventInjector().fail_at(0, 2),
        ]

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, event_injector in zip(range(num_replicas), event_injectors):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=local_sgd_train_loop,
                    use_cuda=use_cuda,
                    manager_args={
                        "use_async_quorum": False,
                    },
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []

            for fut in as_completed(futures):
                try:
                    state_dicts.append(fut.result())
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()

        for state_dict in state_dicts:
            # LocalSGD only guarantees that the model is consistent across
            # replicas but uses separate optimizer states.
            torch.testing.assert_close(
                state_dict[0]["model"], state_dicts[0][0]["model"], check_device=False
            )

        self.assertEqual(event_injectors[1].count[EventInjectorEvent.Failure], 1)

    @parameterized.expand(
        [
            # (True,),
            (False,),
        ]
    )
    def test_diloco_healthy(self, use_cuda: bool) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")
        if sys.platform == "darwin":
            self.skipTest("not reliable on mac")

        lighthouse = LighthouseServer(bind="[::]:0", min_replicas=2)
        num_replicas = 2
        futures = []

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, 1)

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id in range(num_replicas):
                event_injector = EventInjector()
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=diloco_train_loop,
                    use_cuda=use_cuda,
                    train_loop_args={
                        "model_state_dict": m.state_dict(),
                    },
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []
            for fut in as_completed(futures):
                try:
                    state_dicts.append(fut.result()[0])
                except Exception as e:
                    print(e, flush=True)
                    traceback.print_exc()
                    raise

        lighthouse.shutdown()

        rep0, rep1 = state_dicts
        assert_equal_global_state(1, rep1, rep0)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipIf(sys.platform == "darwin", "not reliable on mac")
    @parameterized.expand(
        [
            # (True,),
            (False,),
        ]
    )
    def test_diloco_recovery(self, use_cuda: bool) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")
        if sys.platform == "darwin":
            self.skipTest("not reliable on mac")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        event_injectors = [
            EventInjector(),
            EventInjector().fail_at(0, 2),
        ]

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, 1)

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, event_injector in zip(range(num_replicas), event_injectors):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=diloco_train_loop,
                    train_loop_args={
                        "model_state_dict": m.state_dict(),
                    },
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []

            for fut in as_completed(futures):
                continue

            for fut in futures:
                try:
                    state_dicts.append(fut.result()[0])
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()

        rep0, rep1 = state_dicts

        # Inner optimizer and local model parameters will be different e.g.
        # with 2 replicas r1 and r2, we sync every 2 steps
        #
        # - Manager Step 1
        #   - Step 1: r1 and r2 step
        #   - Step 2: r1 and r2 step, sync the model, quorum succeeds
        # - Manager Step 2
        #   - Step 1: r1 steps but r2 fails
        #   - Step 2:
        #     - r1 steps, sync fails because r2 is down
        #     - r1 recovers r2 from the model state at this step
        #       that is different from the model for r1 at the beginning
        #       of step Manager Step 2
        #
        # Outer optimizer and global model should be the same
        assert_equal_global_state(1, rep1, rep0)

        self.assertEqual(event_injectors[1].count[EventInjectorEvent.Failure], 1)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipIf(sys.platform == "darwin", "not reliable on mac")
    @parameterized.expand(
        [
            # (True,),
            (False,),
        ]
    )
    def test_streaming_diloco_recovery(self, use_cuda: bool) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")
        if sys.platform == "darwin":
            self.skipTest("not reliable on mac")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        event_injectors = [
            EventInjector(),
            EventInjector().fail_at(0, 2),
        ]

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, 2)

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, event_injector in zip(range(num_replicas), event_injectors):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=diloco_train_loop,
                    train_loop_args={
                        "model_state_dict": m.state_dict(),
                        "n_fragments": 2,
                        "diloco_args": {
                            "fragment_sync_delay": 1,
                            "sync_every": 4,
                        },
                    },
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []

            for fut in as_completed(futures):
                continue

            for fut in futures:
                try:
                    state_dicts.append(fut.result()[0])
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()

        rep0, rep1 = state_dicts

        assert_equal_global_state(2, rep1, rep0)

        self.assertEqual(event_injectors[1].count[EventInjectorEvent.Failure], 1)

    CONFIG: list[tuple[bool, int, int, float]] = [
        (use_cuda, n_fragments, fragment_sync_delay, alpha)
        for use_cuda in [False]
        for n_fragments in [1, 2]
        for fragment_sync_delay in [0, 1]
        for alpha in [0.0, 0.5, 1.0]
    ]

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipIf(sys.platform == "darwin", "not reliable on mac")
    @parameterized.expand(CONFIG)
    def test_streaming_diloco_upscale(
        self, use_cuda: bool, n_fragments: int, fragment_sync_delay: int, alpha: float
    ) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")
        if sys.platform == "darwin":
            self.skipTest("not reliable on mac")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 3
        futures = []
        executors = []

        barrier = threading.Barrier(num_replicas)

        event_injectors = [
            # Make this replica join after other replicas have made 2 steps
            EventInjector().barrier_at(0, 0, barrier),
            EventInjector().barrier_at(0, 2, barrier),
            EventInjector().barrier_at(0, 2, barrier),
        ]

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, n_fragments)

        for replica_id, event_injector in zip(range(num_replicas), event_injectors):
            executor = ThreadPoolExecutor(max_workers=1)
            executors.append(executor)
            runner = Runner(
                replica_id=replica_id,
                num_replicas=num_replicas,
                lighthouse_address=lighthouse.address(),
                event_injector=event_injector,
                train_loop=diloco_train_loop,
                train_loop_args={
                    "model_state_dict": m.state_dict(),
                    "n_fragments": n_fragments,
                    "diloco_args": {
                        "fragment_sync_delay": fragment_sync_delay,
                        "sync_every": 4,
                        "fragment_update_alpha": alpha,
                    },
                },
            )
            futures.append(executor.submit(runner.run_replica))

        state_dicts = []

        for fut in as_completed(futures):
            continue

        for fut in futures:
            try:
                state_dicts.append(fut.result()[0])
            except Exception as e:
                print(e)
                raise

        lighthouse.shutdown()

        rep0, rep1, rep2 = state_dicts

        assert_equal_global_state(n_fragments, rep0, rep1)
        assert_equal_global_state(n_fragments, rep0, rep2)

        for event_injector in event_injectors:
            self.assertEqual(event_injectors[1].count[EventInjectorEvent.Barrier], 1)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipIf(sys.platform == "darwin", "not reliable on mac")
    @parameterized.expand(CONFIG)
    def test_streaming_diloco_commit_failure(
        self, use_cuda: bool, n_fragments: int, fragment_sync_delay: int, alpha: float
    ) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")
        if sys.platform == "darwin":
            self.skipTest("not reliable on mac")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []
        executors = []

        event_injectors = [
            EventInjector().fail_allreduce_at(0, 1),
            EventInjector().fail_allreduce_at(0, 1),
        ]

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, n_fragments)

        for replica_id, event_injector in zip(range(num_replicas), event_injectors):
            executor = ThreadPoolExecutor(max_workers=1)
            executors.append(executor)
            runner = Runner(
                replica_id=replica_id,
                num_replicas=num_replicas,
                lighthouse_address=lighthouse.address(),
                event_injector=event_injector,
                train_loop=diloco_train_loop,
                train_loop_args={
                    "model_state_dict": m.state_dict(),
                    "n_fragments": n_fragments,
                    "diloco_args": {
                        "fragment_sync_delay": fragment_sync_delay,
                        "sync_every": 4,
                        "fragment_update_alpha": alpha,
                    },
                },
            )
            futures.append(executor.submit(runner.run_replica))

        state_dicts = []

        for fut in as_completed(futures):
            continue

        for fut in futures:
            try:
                state_dicts.append(fut.result()[0])
            except Exception as e:
                print(e)
                raise

        lighthouse.shutdown()

        rep0, rep1 = state_dicts

        assert_equal_global_state(n_fragments, rep0, rep1)

        for event_injector in event_injectors:
            self.assertEqual(
                event_injector.count[EventInjectorEvent.AllreduceFailure], 1
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
