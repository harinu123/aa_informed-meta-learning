import argparse
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


AA_RESNAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]


def _get_label(sample):
    if "label" in sample:
        return float(sample["label"])
    if "labels" in sample:
        labels = sample["labels"]
        for key in ("affinity", "pK", "pKd", "pKi", "label"):
            if key in labels:
                return float(labels[key])
    raise KeyError("No affinity label found in ATOM3D LBA sample.")


def _get_protein_id(sample):
    for key in ("protein_id", "pdb_id", "id", "pdb"):
        if key in sample:
            return str(sample[key])
    return "unknown_protein"


def _split_atoms(atoms_df):
    if "is_ligand" in atoms_df.columns:
        ligand_atoms = atoms_df[atoms_df["is_ligand"]]
        protein_atoms = atoms_df[~atoms_df["is_ligand"]]
        return protein_atoms, ligand_atoms

    if "atom_type" in atoms_df.columns:
        ligand_atoms = atoms_df[atoms_df["atom_type"].str.lower() == "ligand"]
        protein_atoms = atoms_df[atoms_df["atom_type"].str.lower() != "ligand"]
        return protein_atoms, ligand_atoms

    if "chain" in atoms_df.columns:
        ligand_atoms = atoms_df[atoms_df["chain"].isin(["L", "l"])]
        protein_atoms = atoms_df[~atoms_df["chain"].isin(["L", "l"])]
        if len(ligand_atoms) > 0:
            return protein_atoms, ligand_atoms

    if "resname" in atoms_df.columns:
        ligand_atoms = atoms_df[~atoms_df["resname"].isin(AA_RESNAMES)]
        protein_atoms = atoms_df[atoms_df["resname"].isin(AA_RESNAMES)]
        return protein_atoms, ligand_atoms

    raise ValueError("Unable to split protein and ligand atoms from sample.")


def _extract_coords(atoms_df):
    return atoms_df[["x", "y", "z"]].to_numpy(dtype=np.float32)


def _get_protein_ca_coords(protein_atoms):
    if "atom_name" in protein_atoms.columns:
        ca_atoms = protein_atoms[protein_atoms["atom_name"] == "CA"]
        if len(ca_atoms) >= 3:
            return _extract_coords(ca_atoms)
    coords = _extract_coords(protein_atoms)
    if coords.shape[0] < 3:
        return None
    return coords


def _ligand_centroid(ligand_atoms):
    if "element" in ligand_atoms.columns:
        heavy = ligand_atoms[ligand_atoms["element"] != "H"]
        if len(heavy) > 0:
            return _extract_coords(heavy).mean(axis=0)
    coords = _extract_coords(ligand_atoms)
    if coords.shape[0] == 0:
        return None
    return coords.mean(axis=0)


def _ligand_features(ligand_atoms):
    elements = ligand_atoms["element"].tolist() if "element" in ligand_atoms.columns else []
    counts = Counter(elements)
    feat = [counts.get(el, 0) for el in ELEMENTS]

    coords = _extract_coords(ligand_atoms)
    if coords.shape[0] == 0:
        mean_dist = 0.0
        max_dist = 0.0
    else:
        centroid = coords.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(coords - centroid, axis=1)
        mean_dist = float(dists.mean())
        max_dist = float(dists.max())

    num_heavy = int(sum(1 for el in elements if el != "H"))
    feat.extend([num_heavy, mean_dist, max_dist])
    return np.array(feat, dtype=np.float32)


def _pca_delta(protein_ca_coords, ligand_centroid):
    center = protein_ca_coords.mean(axis=0)
    pca = PCA(n_components=3)
    pca.fit(protein_ca_coords - center)
    delta = (ligand_centroid - center) @ pca.components_.T
    return delta.astype(np.float32)


def _load_atom3d_lba(data_dir=None):
    try:
        from atom3d.datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "ATOM3D is not installed. Install with `pip install atom3d`."
        ) from exc

    if data_dir is None:
        return load_dataset("lba")
    try:
        return load_dataset("lba", data_dir=data_dir)
    except TypeError:
        return load_dataset("lba", data_dir)


def _iter_records(dataset):
    if isinstance(dataset, dict):
        for split in dataset:
            for record in dataset[split]:
                yield record
    else:
        for record in dataset:
            yield record


def build_tasks(
    dataset,
    num_proteins,
    k_pockets,
    episode_size,
):
    protein_entries = defaultdict(list)

    for record in _iter_records(dataset):
        atoms = record["atoms"]
        protein_atoms, ligand_atoms = _split_atoms(atoms)

        protein_ca = _get_protein_ca_coords(protein_atoms)
        if protein_ca is None:
            continue

        ligand_centroid = _ligand_centroid(ligand_atoms)
        if ligand_centroid is None:
            continue

        delta = _pca_delta(protein_ca, ligand_centroid)
        label = _get_label(record)
        protein_id = _get_protein_id(record)
        features = _ligand_features(ligand_atoms)

        protein_entries[protein_id].append(
            {
                "delta": delta,
                "features": features,
                "label": label,
            }
        )

    proteins_sorted = sorted(
        protein_entries.keys(), key=lambda k: len(protein_entries[k]), reverse=True
    )
    proteins_sorted = proteins_sorted[:num_proteins]

    tasks_by_protein = {}
    for protein_id in proteins_sorted:
        entries = protein_entries[protein_id]
        deltas = np.stack([e["delta"] for e in entries], axis=0)
        num_clusters = min(k_pockets, deltas.shape[0])
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=0)
        labels = kmeans.fit_predict(deltas)
        centers = kmeans.cluster_centers_

        tasks = defaultdict(list)
        for entry, pocket_id in zip(entries, labels):
            tasks[pocket_id].append(entry)

        protein_tasks = []
        for pocket_id, pocket_entries in tasks.items():
            X = np.stack([e["features"] for e in pocket_entries], axis=0)
            Y = np.array([[e["label"]] for e in pocket_entries], dtype=np.float32)
            pocket_center = centers[pocket_id]
            k_vec = np.concatenate(
                [pocket_center, np.array([np.linalg.norm(pocket_center)], dtype=np.float32)]
            )
            protein_tasks.append(
                {
                    "protein_id": protein_id,
                    "pocket_id": int(pocket_id),
                    "X": X.astype(np.float32),
                    "Y": Y,
                    "k": k_vec.astype(np.float32),
                }
            )

        tasks_by_protein[protein_id] = protein_tasks

    train_val_proteins = proteins_sorted[:-1]
    test_proteins = proteins_sorted[-1:]

    train_tasks = []
    val_tasks = []
    test_tasks = []
    for protein_id in train_val_proteins:
        tasks = tasks_by_protein[protein_id]
        split_idx = int(0.8 * len(tasks))
        train_tasks.extend(tasks[:split_idx])
        val_tasks.extend(tasks[split_idx:])
    for protein_id in test_proteins:
        test_tasks.extend(tasks_by_protein[protein_id])

    def _filter_tasks(task_list):
        return [
            task
            for task in task_list
            if task["X"].shape[0] >= episode_size
        ]

    train_tasks = _filter_tasks(train_tasks)
    val_tasks = _filter_tasks(val_tasks)
    test_tasks = _filter_tasks(test_tasks)

    return train_tasks, val_tasks, test_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/atom3d-lba-pocket-poc")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-proteins", type=int, default=3)
    parser.add_argument("--k-pockets", type=int, default=2)
    parser.add_argument("--episode-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = _load_atom3d_lba(args.data_dir)
    train_tasks, val_tasks, test_tasks = build_tasks(
        dataset,
        num_proteins=args.num_proteins,
        k_pockets=args.k_pockets,
        episode_size=args.episode_size,
    )

    if not train_tasks and not test_tasks:
        raise RuntimeError("No tasks were created. Check ATOM3D LBA data parsing.")

    x_dim = train_tasks[0]["X"].shape[1]
    meta = {
        "x_dim": x_dim,
        "k_dim": 4,
        "episode_size": args.episode_size,
        "k_pockets": args.k_pockets,
        "num_proteins": args.num_proteins,
    }

    if args.dry_run:
        def _stats(tasks):
            lengths = [t["X"].shape[0] for t in tasks]
            return (
                len(tasks),
                int(np.min(lengths)) if lengths else 0,
                int(np.median(lengths)) if lengths else 0,
            )

        train_stats = _stats(train_tasks)
        val_stats = _stats(val_tasks)
        test_stats = _stats(test_tasks)

        print(f"Proteins selected: {args.num_proteins}")
        print(f"Train tasks: {train_stats[0]}, min/median points: {train_stats[1]}/{train_stats[2]}")
        print(f"Val tasks: {val_stats[0]}, min/median points: {val_stats[1]}/{val_stats[2]}")
        print(f"Test tasks: {test_stats[0]}, min/median points: {test_stats[1]}/{test_stats[2]}")
        print(f"x_dim={x_dim}, k_dim=4, episode_size={args.episode_size}")
        return

    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "tasks.pt")
    torch.save(
        {
            "meta": meta,
            "train": train_tasks,
            "val": val_tasks,
            "test": test_tasks,
        },
        output_path,
    )
    print(f"Saved tasks to {output_path}")
    print(meta)


if __name__ == "__main__":
    main()
