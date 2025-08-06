# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from datetime import timedelta

import torch
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter

from torchft import Manager, ProcessGroupGloo
from torchft.checkpointing.pg_transport import PGTransport
from torchft.local_sgd import DesLoc

logging.basicConfig(level=logging.INFO)


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    RUN = int(os.environ.get("RUN", 0))

    output_folder = f"output/replica-{REPLICA_GROUP_ID}"

    writer = SummaryWriter(f"{output_folder}/tensorboard", max_queue=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pg = ProcessGroupGloo(timeout=timedelta(seconds=5))

    transport = PGTransport(
        pg,
        timeout=timedelta(seconds=10),
        device=device,
    )

    class SimpleModel(nn.Module):
        def __init__(self, d_hid: int, num_classes: int = 10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, d_hid),
                nn.ReLU(),
                nn.Linear(d_hid, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    d_hid = 128
    m = SimpleModel(d_hid).to(device)
    optimizer = optim.AdamW(m.parameters(), lr=1e-3)

    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])

    def state_dict():
        return {
            "model": m.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

    manager = Manager(
        pg=pg,
        use_async_quorum=False,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"train_desloc_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=30),
        checkpoint_transport=transport,
    )

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, feature_dim=128, num_classes=10):
            self.size = size
            self.feature_dim = feature_dim
            self.num_classes = num_classes

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            features = torch.rand(self.feature_dim)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return features, label

    trainset = DummyDataset(feature_dim=d_hid)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, num_workers=2, shuffle=True
    )

    criterion = nn.CrossEntropyLoss()

    tensorboard_key_prefix = f"Run:{RUN}"
    with DesLoc(
        manager,
        m,
        optimizer,
        param_sync_every=10,
        optimizer_sync_every=[20, 20],  # For AdamW's exp_avg and exp_avg_sq
    ) as desloc:
        while True:
            for i, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                out = m(inputs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                writer.add_scalar(f"{tensorboard_key_prefix}/loss", loss, i)
                writer.add_scalar(
                    f"{tensorboard_key_prefix}/num_participants",
                    manager.num_participants(),
                    i,
                )
                writer.add_scalar(
                    f"{tensorboard_key_prefix}/current_step", manager.current_step(), i
                )
                if manager.current_step() % 10 == 0:
                    print(f"[{manager.current_step()}] loss = {loss.item()}")

                if manager.current_step() >= 100:
                    # complete training
                    writer.flush()
                    exit()


if __name__ == "__main__":
    main()
