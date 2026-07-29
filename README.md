# MetaGIN: A Lightweight Framework for Molecular Property Prediction

**MetaGIN** is a compact graph neural network for molecular property prediction.
It extends GIN-style message passing with **1-/2-/3-hop edges** (bonds, bond
angles, torsion angles), a **virtual node**, and **MetaFormer** residual mixers,
targeting high accuracy at under 10M parameters.

On [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) (OGB-LSC), MetaGIN
reaches **0.0851** validation MAE with about **8.87M** parameters, and reports
strong results on several [MoleculeNet](https://moleculenet.org/) benchmarks.

## Publication

Xuan Zhang, Cheng Chen, Xiaoting Wang, Haitao Jiang, Wei Zhao, and Xuefeng Cui.
*MetaGIN: a lightweight framework for molecular property prediction.*
*Frontiers of Computer Science*, 19:195912, 2025.

- **Springer:** https://link.springer.com/article/10.1007/s11704-024-3784-y
- **DOI:** https://doi.org/10.1007/s11704-024-3784-y
- **HEP journal page:** https://journal.hep.com.cn/fcs/EN/10.1007/s11704-024-3784-y

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

## Method (short)

| Level | Block | Role |
|---|---|---|
| Local | `ConvKernel` / `ConvBlock` | Multi-hop aggregation on bond / angle / torsion edges |
| Global | `VirtKernel` | Graph-level virtual node with recurrent residual state |
| Mix | `MetaFormerBlock` + `GatedLinearBlock` | Scaled residual + GLU mixer (MetaFormer-style) |
| Optional | `KnnKernel` | FAISS kNN edges + train-time distance auxiliary (last layer) |
| Head | `HeadBlock` | Virtual + node pooled regression to HOMO–LUMO gap |

Implementation lives under [`model/`](model/). Architecture: [`model/model.py`](model/model.py);
dataset / multi-hop featurization: [`model/data.py`](model/data.py); training:
[`model/main.py`](model/main.py). See [`model/README.md`](model/README.md) for
setup, preprocessing, and config naming (`tiny311`, etc.).

Predecessor in this line: [CoAtGIN](https://github.com/xfcui/CoAtGIN)
([IEEE BIBM 2022](https://ieeexplore.ieee.org/document/9995324/)).

## License

[MIT](LICENSE) — Copyright (c) 2022 Xuefeng Cui.
