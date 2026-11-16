import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
import pytorch_lightning as pl
from torch_geometric.data import Data, InMemoryDataset

ATOM_ENCODER = {"C": 0, "N": 1, "O": 2}
ATOM_DECODER = ["C", "N", "O"]
N_ATOM_TYPES = 3
N_BOND_TYPES = 5
Y_DIM = 6


class AOConditionalDataset(InMemoryDataset):

    def __init__(self, split, root, stats, p_uncond=0.0, transform=None):
        self.split = split
        self.stats = stats
        self.p_uncond = p_uncond
        super().__init__(root, transform)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path, weights_only=False)
        self._smiles = [d.smiles for d in self]

    @property
    def raw_file_names(self):
        return [f"{self.split}.pt"]

    @property
    def processed_file_names(self):
        return [f"{self.split}_ao_cond.pt"]

    def download(self):
        pass

    def _z(self, col, val):
        if col not in self.stats:
            return float(val)
        mu, sigma = (self.stats[col]["mean"], self.stats[col]["std"])
        return (float(val) - mu) / (sigma + 1e-08)

    def _make_y(self, raw_y):
        type_p = raw_y[0, 0].item()
        type_a = raw_y[0, 1].item()
        bde_z = self._z("bde", raw_y[0, 2].item())
        logp_z = self._z("logp", raw_y[0, 3].item())
        sa_z = self._z("sa_score", raw_y[0, 4].item())
        syba_z = self._z("syba", raw_y[0, 5].item())
        return torch.tensor(
            [[type_p, type_a, bde_z, logp_z, sa_z, syba_z]], dtype=torch.float
        )

    def process(self):
        raw_data = torch.load(self.raw_paths[0], weights_only=False)
        data_list = []
        skipped = 0
        for d in raw_data:
            if not hasattr(d, "y") or d.y is None:
                skipped += 1
                continue
            d.y = self._make_y(d.y)
            data_list.append(d)
        if skipped:
            print(f"  [{self.split}] Skipped {skipped} items without y")
        print(f"  [{self.split}] {len(data_list):,} molecules")
        torch.save(self.collate(data_list), self.processed_paths[0])

    @property
    def smiles(self):
        return set((getattr(d, "smiles", "") for d in self))


def _compute_statistics(dataset):
    from collections import Counter

    node_counts = Counter()
    node_type_counts = torch.zeros(N_ATOM_TYPES)
    edge_type_counts = torch.zeros(N_BOND_TYPES)
    valency_counts = torch.zeros(200)
    for data in dataset:
        n = data.x.shape[0]
        node_counts[n] += 1
        node_type_counts += data.x.sum(dim=0)
        if data.edge_index.numel() > 0:
            edge_type_counts[1:] += data.edge_attr[:, 1:].sum(dim=0)
            n_pairs = n * (n - 1)
            n_edges = data.edge_index.shape[1]
            edge_type_counts[0] += max(0, n_pairs - n_edges)
    max_n = max(node_counts.keys()) + 1
    node_count_tensor = torch.zeros(max_n)
    for k, v in node_counts.items():
        node_count_tensor[k] = v
    return (node_count_tensor, node_type_counts, edge_type_counts)


class AOConditionalDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        from hydra.utils import get_original_cwd
        import pathlib

        base = pathlib.Path(get_original_cwd()).parents[1]
        root = os.path.join(base, cfg.dataset.datadir)
        stats_p = pathlib.Path(root) / "raw" / "condition_stats.json"
        stats = json.load(open(stats_p)) if stats_p.exists() else {}
        p_uncond = getattr(cfg.dataset, "p_uncond", 0.0)
        self.train_dataset = AOConditionalDataset("train", root, stats, p_uncond)
        self.val_dataset = AOConditionalDataset("val", root, stats, p_uncond=0.0)
        self.test_dataset = AOConditionalDataset("test", root, stats, p_uncond=0.0)
        self.cfg = cfg
        bs = cfg.train.batch_size
        nw = getattr(cfg.train, "num_workers", 0)
        pin = getattr(cfg.dataset, "pin_memory", False)
        from torch_geometric.loader import DataLoader

        self._train_loader = DataLoader(
            self.train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
        )
        self._val_loader = DataLoader(
            self.val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
        )
        self._test_loader = DataLoader(
            self.test_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
        )
        n_cnt, n_typ, e_typ = _compute_statistics(self.train_dataset)
        self.statistics = {
            "train": _make_stats_obj(n_cnt, n_typ, e_typ),
            "val": _make_stats_obj(*_compute_statistics(self.val_dataset)),
            "test": _make_stats_obj(*_compute_statistics(self.test_dataset)),
        }

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader


class _Stats:

    def __init__(self, num_nodes, node_types, edge_types):
        self.num_nodes = num_nodes
        self.node_types = node_types
        self.edge_types = edge_types


def _make_stats_obj(n_cnt, n_typ, e_typ):
    return _Stats(n_cnt, n_typ / n_typ.sum(), e_typ / e_typ.sum())


class AOConditionalInfos:

    def __init__(self, datamodule, cfg):
        from src.diffusion.distributions import DistributionNodes
        import src.utils as utils

        self.name = "ao_conditional"
        self.is_molecular = True
        self.remove_h = True
        self.use_charge = False
        self.atom_encoder = ATOM_ENCODER
        self.atom_decoder = ATOM_DECODER
        self.statistics = datamodule.statistics
        self.train_smiles = datamodule.train_dataset.smiles
        self.val_smiles = datamodule.val_dataset.smiles
        self.test_smiles = datamodule.test_dataset.smiles
        train_stats = datamodule.statistics["train"]
        self.node_types = train_stats.node_types
        self.edge_types = train_stats.edge_types
        n_nodes_dist = train_stats.num_nodes
        self.nodes_dist = DistributionNodes(n_nodes_dist)
        self.max_n_nodes = len(n_nodes_dist) - 1
        self.num_classes = N_ATOM_TYPES
        self.input_dims = None
        self.output_dims = None
        self.valencies = [4, 3, 2]
        self.atom_weights = [12, 14, 16]
        self.max_weight = 2000

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        import src.utils as utils

        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch,
        )
        example_data = {
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": example_batch["y"],
            "node_mask": node_mask,
        }
        self.input_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": example_batch["y"].size(1) + 1,
        }
        ex_extra = extra_features(example_data)
        self.input_dims["X"] += ex_extra.X.size(-1)
        self.input_dims["E"] += ex_extra.E.size(-1)
        self.input_dims["y"] += ex_extra.y.size(-1)
        ex_mol = domain_features(example_data)
        self.input_dims["X"] += ex_mol.X.size(-1)
        self.input_dims["E"] += ex_mol.E.size(-1)
        self.input_dims["y"] += ex_mol.y.size(-1)
        self.output_dims = {
            "X": example_batch["x"].size(1),
            "E": example_batch["edge_attr"].size(1),
            "y": 0,
        }
