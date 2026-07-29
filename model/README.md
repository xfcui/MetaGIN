# Training MetaGIN

PyTorch / PyG implementation of **MetaGIN** for
[PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/).
Paper: [Springer](https://link.springer.com/article/10.1007/s11704-024-3784-y)
([DOI](https://doi.org/10.1007/s11704-024-3784-y)).

MetaGIN builds on [MetaFormer](https://arxiv.org/abs/2210.13452) and
[GIN](https://openreview.net/forum?id=ryGs6iA5Km). Beyond 1-hop covalent bonds,
it materializes **2-hop (angle)** and **3-hop (torsion)** edges so the graph
carries coarse 3D geometry without continuous coordinates at inference.

## Requirements

Typical stack:

- Python 3, CUDA GPU
- `torch`, `torch-geometric`, `torch-scatter`, `torch-sparse`
- `ogb`, `rdkit`, `h5py`, `pandas`, `tqdm`, `numpy`
- `faiss-gpu` (for the last-layer `KnnKernel`)

Also required under `model/data/`: `atom2feat.hdf` (atom feature lookup used at
load time).

Run scripts from this directory (`model/`) so relative imports and the
`data/` cache root resolve correctly.

## Preprocess

Build the MetaGIN heterogeneous PCQM4Mv2 cache (bond / angle / torsion edges,
RWPE, optional 3D positions):

```bash
cd model
# expects atom2feat.hdf under ./data/; writes under ./data/pcqm4m-metagin/
python3 -BuW ignore data.py
```

Or use [`data.sh`](data.sh) (run from the workspace layout that owns `data/`,
or edit paths). First run downloads OGB PCQM4Mv2 if needed.

## Train

Default compact config (`tiny311`: hop=3, kernel×1, virtual node on):

```bash
cd model
python3 -BuW ignore main.py --model tiny311 --save .
```

| Flag | Default | Notes |
|---|---|---|
| `--model` | `tiny311` | Size + hop/kernel/virt digits (table below) |
| `--save` | `''` | If set, saves `modelXXX.pt` each epoch and test-dev submissions |

Defaults in [`main.py`](main.py): batch **256**, peak LR **3e-3**, WD **2e-2**,
**12×12** cosine periods (144 epochs), Adan + custom `Scheduler` in
[`optim.py`](optim.py). Train loss = gap L1 + distance SmoothL1 / 4 (when the
last layer is `KnnKernel`).

## Config names (`--model`)

Pattern: `{size}{hop}{kernel}{virt}`

| Part | Values | Meaning |
|---|---|---|
| `size` | `tiny` / `base` / `large` | Per-layer VoVNet kernel schedule (depth 4, 16 heads) |
| `hop` | `1`–`3` | Max hop used in `ConvKernel` (bond / +angle / +torsion) |
| `kernel` | ≥`1` | Multiplier on the size’s per-layer kernel counts |
| `virt` | `0` / `1` | Disable / enable `VirtKernel` |

| `--model` | hop | kernel× | virt | Role |
|---|---:|---:|:---:|---|
| **`tiny311`** | **3** | **1** | **✓** | Compact default |
| `base321` | 3 | 2 | ✓ | Wider kernel schedule |
| `large321` | 3 | 2 | ✓ | Largest kernel schedule |

`tiny` / `base` / `large` set `conv_kernel` base lists
`[1,1,1,1]` / `[1,1,3,1]` / `[1,4,9,1]` before the digit multiplier. The last
layer always uses `KnnKernel` instead of `ConvKernel` in the checked-in code.

## Layout

| File | Role |
|---|---|
| [`model.py`](model.py) | `MetaGIN`, conv / virt / knn / MetaFormer / head |
| [`data.py`](data.py) | Multi-hop featurization + `PygPCQM4Mv2Dataset` |
| [`main.py`](main.py) | Train / val / test-dev loop |
| [`optim.py`](optim.py) | Param groups + warmup / cosine `Scheduler` |
| [`adan.py`](adan.py) | Adan optimizer (Apache-2.0, Garena) |
| [`main.sh`](main.sh) / [`data.sh`](data.sh) | Example multi-GPU / preprocess launchers |

## Citation

```bibtex
@article{Zhang2025MetaGIN,
  author  = {Zhang, Xuan and Chen, Cheng and Wang, Xiaoting and Jiang, Haitao
             and Zhao, Wei and Cui, Xuefeng},
  title   = {{MetaGIN}: a lightweight framework for molecular property prediction},
  journal = {Frontiers of Computer Science},
  year    = {2025},
  volume  = {19},
  number  = {5},
  pages   = {195912},
  doi     = {10.1007/s11704-024-3784-y},
  url     = {https://link.springer.com/article/10.1007/s11704-024-3784-y}
}
```
