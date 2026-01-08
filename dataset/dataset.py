import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

ROOT = "data/"


class SetKnowledgeTrendingSinusoids(Dataset):
    def __init__(
        self,
        split="train",
        root=f"{ROOT}/trending-sinusoids",
        knowledge_type="full",
        split_file="splits",
    ):
        self.data = pd.read_csv(f"{root}/data.csv")
        self.knowledge = pd.read_csv(f"{root}/knowledge.csv")
        self.value_cols = [c for c in self.data.columns if c.isnumeric()]
        self.dim_x = 1
        self.dim_y = 1
        if split_file is None:
            split_file = "splits"
        self.train_test_val_split = pd.read_csv(f"{root}/{split_file}.csv")
        self.split = split
        self.knowledge_type = knowledge_type
        self.knowledge_input_dim = 4

        self._split_data()

    def _split_data(self):
        if self.split == "train":
            train_ids = self.train_test_val_split[
                self.train_test_val_split["split"] == "train"
            ].curve_id
            self.data = self.data[self.data.curve_id.isin(train_ids)]
        elif self.split == "val" or self.split == "valid":
            val_ids = self.train_test_val_split[
                self.train_test_val_split["split"] == "val"
            ].curve_id
            self.data = self.data[self.data.curve_id.isin(val_ids)]
        elif self.split == "test":
            test_ids = self.train_test_val_split[
                self.train_test_val_split["split"] == "test"
            ].curve_id
            self.data = self.data[self.data.curve_id.isin(test_ids)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y = self.data.iloc[idx, :][self.value_cols].values
        x = np.linspace(-2, 2, len(y))
        curve_id = self.data.iloc[idx]["curve_id"]

        knowledge = self.get_knowledge(curve_id)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(
            -1
        )  # [bs,  num_points, x_size]
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(
            -1
        )  # [bs,  num_points, x_size]

        return x, y, knowledge

    def get_knowledge(self, curve_id):
        knowledge = (
            self.knowledge[self.knowledge.curve_id == curve_id]
            .drop("curve_id", axis=1)
            .values
        )
        knowledge = torch.tensor(knowledge, dtype=torch.float32).reshape(3, 1)
        indicator = torch.eye(3)
        knowledge = torch.cat([indicator, knowledge], dim=1)

        if self.knowledge_type == "abc":
            revealed = np.random.choice([0, 1, 2])
            mask = torch.zeros((3, 1))
            mask[revealed] = 1.0
            knowledge = knowledge * mask
        elif self.knowledge_type == "abc2":
            mask = torch.zeros((3, 1))
            num_revealed = np.random.choice([1, 2])
            revealed = np.random.choice([0, 1, 2], num_revealed, replace=False)
            mask[revealed] = 1.0
            knowledge = knowledge * mask
        elif self.knowledge_type == "a":
            knowledge = knowledge[0, :].unsqueeze(0)
        elif self.knowledge_type == "b":
            knowledge = knowledge[1, :].unsqueeze(0)
        elif self.knowledge_type == "c":
            knowledge = knowledge[2, :].unsqueeze(0)
        elif self.knowledge_type == "full":
            pass
        elif self.knowledge_type == "none":
            knowledge = torch.zeros_like(knowledge)
        else:
            raise NotImplementedError

        return knowledge


class SetKnowledgeTrendingSinusoidsDistShift(SetKnowledgeTrendingSinusoids):
    def __init__(
        self,
        split="train",
        root="./data/trending-sinusoids-dist-shift",
        knowledge_type="full",
        split_file="splits",
    ):
        super().__init__(
            split=split, root=root, knowledge_type=knowledge_type, split_file=split_file
        )


class Temperatures(Dataset):
    def __init__(
        self, split="train", root="./data/temperatures", knowledge_type="min_max"
    ):
        region = "AK"
        self.data = pd.read_csv(f"{root}/2021-2022_{region}.csv")
        self.splits = pd.read_csv(f"{root}/2021-2022_{region}_splits.csv")
        if knowledge_type == "desc":
            self.knowledge_df = pd.read_csv(
                f"{root}/2021-2022_{region}_gpt_descriptions.csv"
            )
        elif knowledge_type in ["min_max", "min_max_month"]:
            self.knowledge_df = pd.read_csv(f"{root}/2021-2022_{region}_knowledge.csv")
        elif knowledge_type == "llama_embed":
            knowledge_ds = load_from_disk(
                f"{root}/2021-2022_{region}_desc-embeded-llama"
            )
            self.knowledge_df = knowledge_ds.to_pandas()

        self.knowledge_type = knowledge_type
        if knowledge_type == "min_max":
            self.knowledge_input_dim = 2
        elif knowledge_type == "min_max_month":
            self.knowledge_input_dim = 3
        elif knowledge_type == "desc":
            self.knowledge_input_dim = None
        elif knowledge_type == "llama_embed":
            self.knowledge_input_dim = 4096
        else:
            raise NotImplementedError

        self.split = split
        if self.split == "train":
            dates = self.splits[self.splits.split == "train"].LST_DATE
        elif self.split == "val" or self.split == "valid":
            dates = self.splits[self.splits.split == "val"].LST_DATE
        elif self.split == "test":
            dates = self.splits[self.splits.split == "test"].LST_DATE

        self.data = self.data[self.data.LST_DATE.isin(dates)]

        self.dim_x = 1
        self.dim_y = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y = self.data.iloc[idx, 1:].values
        lst_date = self.data.iloc[idx, 0]
        k_idx = self.knowledge_df[self.knowledge_df.LST_DATE == lst_date].index[0]
        x = np.linspace(-2, 2, len(y))
        if self.knowledge_type == "min_max":
            knowledge = self.knowledge_df[["min", "max"]].iloc[k_idx, :].values
            knowledge = torch.tensor(knowledge, dtype=torch.float32)
        elif self.knowledge_type == "min_max_month":
            knowledge = self.knowledge_df[["min", "max", "month"]].iloc[k_idx, :].values
            knowledge = torch.tensor(knowledge, dtype=torch.float32)
        elif self.knowledge_type == "desc":
            knowledge = self.knowledge_df.iloc[k_idx, :].description
        elif self.knowledge_type == "llama_embed":
            knowledge = self.knowledge_df.iloc[k_idx, :].embed[0]
            knowledge = torch.tensor(knowledge)
        else:
            raise NotImplementedError

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        return x, y, knowledge


class Atom3DLBAPocketPOC(Dataset):
    def __init__(self, split="train", root="./data/atom3d-lba-pocket-poc", episode_size=None):
        path = os.path.join(root, "tasks.pt")
        blob = torch.load(path)
        meta = blob["meta"]
        self.tasks = blob[split]
        self.split = split
        self.dim_x = meta["x_dim"]
        self.dim_y = 1
        self.knowledge_input_dim = meta["k_dim"]
        self.episode_size = episode_size or meta["episode_size"]
        self._filter_tasks()

    def _filter_tasks(self):
        self.tasks = [
            task
            for task in self.tasks
            if task["X"].shape[0] >= self.episode_size
        ]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        X = task["X"]
        Y = task["Y"]
        k = task["k"]

        if X.shape[0] >= self.episode_size:
            indices = np.random.choice(X.shape[0], self.episode_size, replace=False)
        else:
            indices = np.random.choice(X.shape[0], self.episode_size, replace=True)

        x = torch.tensor(X[indices], dtype=torch.float32)
        y = torch.tensor(Y[indices], dtype=torch.float32)
        knowledge = torch.tensor(k, dtype=torch.float32)

        return x, y, knowledge


class ModularAdditionRotations(Dataset):
    def __init__(
        self,
        split="train",
        p=113,
        m_train_max=20,
        m_test_min=21,
        m_test_max=40,
        episode_size=2048,
        seed=0,
        knowledge_type="w",
    ):
        self.split = split
        self.p = p
        self.m_train_max = m_train_max
        self.m_test_min = m_test_min
        self.m_test_max = m_test_max
        self.episode_size = episode_size
        self.seed = seed
        self.knowledge_type = knowledge_type
        self.dim_x = 2
        self.dim_y = 2
        self.knowledge_input_dim = 1

        if split == "train":
            self.length = 5000
            self.split_offset = 0
        elif split in ("val", "valid"):
            self.length = 1000
            self.split_offset = 10_000
        elif split == "test":
            self.length = 1000
            self.split_offset = 20_000
        else:
            raise ValueError(f"Unknown split {split}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx + self.split_offset)

        if self.split == "test":
            m = rng.integers(self.m_test_min, self.m_test_max + 1)
        else:
            m = rng.integers(1, self.m_train_max + 1)

        a = rng.integers(0, self.p, size=self.episode_size)
        b = rng.integers(0, self.p, size=self.episode_size)
        c = (a + b) % self.p

        a_norm = 2 * a / (self.p - 1) - 1
        b_norm = 2 * b / (self.p - 1) - 1
        x = np.stack([a_norm, b_norm], axis=-1).astype(np.float32)

        w = 2 * np.pi * m / self.p
        y = np.stack([np.sin(w * c), np.cos(w * c)], axis=-1).astype(np.float32)

        if self.knowledge_type == "w":
            knowledge = torch.tensor([[w]], dtype=torch.float32)
        elif self.knowledge_type == "none":
            knowledge = torch.zeros((1, 1), dtype=torch.float32)
        else:
            raise NotImplementedError

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y, knowledge
