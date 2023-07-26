# Sparse representation for machine learning the properties of defects in 2D materials
# Quickstart
Open in Constructor Research Platform (a cloud service for scientific computations)

[![Open in Constructor Research](docs/research_platform_badge.svg)](https://research.constructor.tech/p/2d-defects-prediction)

# Table of contents
- Environment setup: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
- Data download: [docs/DATA.md](docs/DATA.md)
- Reproducing the paper locally: [docs/SPARSE-PAPER.md](docs/SPARSE-PAPER.md)
- [Using the pre-trained models](#using-the-pre-trained-models)

# Summary
In the paper we propose sparse representation as a way to reduce the computational cost and improve the accuracy of machine learning the properties of defects in 2D materials. The code in the project implements the method, and a rigorous comparison of its performance to the a set of baselines.

Two-dimensional materials offer a promising platform for the next generation of (opto-) electronic devices and other high technology applications. One of the most exciting characteristics of 2D crystals is the ability to tune their properties via controllable introduction of defects. However, the search space for such structures is enormous, and ab-initio computations prohibitively expensive. We propose a machine learning approach for rapid estimation of the properties of 2D material given the lattice structure and defect configuration. The method suggests a way to represent  configuration of 2D materials with defects that allows a neural network to train quickly and accurately. We compare our methodology with the state-of-the-art approaches and demonstrate at least 3.7 times energy prediction error drop. Also, our approach is an order of magnitude more resource-efficient than its contenders both for the training and inference part.

The main idea of our method is using a point cloud of defects as an input to the predictive model, as opposed to the usual point cloud of atoms, or expertly created feature vector.
![Sparse representation construction](./docs/constructor_pics/sparse.png)

We compare our approach to state-of-the-art generic structure-property prediction algorithms: [GemNet](https://arxiv.org/abs/2106.08903), [SchNet](https://arxiv.org/abs/1706.08566), [MegNet](https://arxiv.org/abs/1812.05055), [matminer+CatBoost](https://github.com/hackingmaterials/matminer).

For dataset, we use [2DMD](https://www.nature.com/articles/s41699-023-00369-1). It consists of the most popular 2D materials: MoS2, WSe2, h-BN, GaSe, InSe, and black phosphorous (BP) with point defect density in the range of 2.5% to 12.5%. We use DFT to relax the structures and compute the defect formation energy and HOMO-LUMO gap. ML algorithms predict those quantities, taking unrelaxed structures as input.
# Using the pre-trained models
## Library
Use the library https://github.com/HSE-LAMBDA/MEGNetSparse/
## This repository
1. Clone the repository
2. [Set up the environment](docs/ENVIRONMENT.md)
3. Download the weights and data:
```bash
dvc pull datasets/checkpoints/combined_mixed_all_train/formation_energy_per_site/megnet_pytorch/sparse/05-12-2022_19-50-53/d6b7ce45/0.pth.dvc datasets/checkpoints/combined_mixed_all_train/homo_lumo_gap_min/megnet_pytorch/sparse/05-12-2022_19-50-53/831cc496/0.pth.dvc csv-cif-low-density-8x8 csv-cif-no-spin-500-data csv-cif-spin-500-data train-only-split
```
The data are not needed for predictions, and are only used to generate new structures in the example notebook.

4. Open the [notebook](./notebooks/Inference.ipynb). It contains the prediction code, along with generation of new structures with defects, and example processing of user-uploaded data.

# Citation
Please cite the following two papers if you use the code or the data:
```
Kazeev, N., Al-Maeeni, A.R., Romanov, I. et al. Sparse representation for machine learning the properties of defects in 2D materials. npj Comput Mater 9, 113 (2023). https://doi.org/10.1038/s41524-023-01062-z
```

```
Huang, P., Lukin, R., Faleev, M. et al. Unveiling the complex structure-property correlation of defects in 2D materials based on high throughput datasets. npj 2D Mater Appl 7, 6 (2023). https://doi.org/10.1038/s41699-023-00369-1
```
# Internal links
- The overall design is documented in an obsolete [flowchart](https://miro.com/welcomeonboard/eUdTWFNlaTZOZkc3NUlqd2o0TXB2QUUxRjFWVGxVcGtrWTJ5U01lbFZ1aFZxTFJRcUNyNG5NMjFaZkZ4S3pHRXwzMDc0NDU3MzU5MDMzOTQ0ODgx?invite_link_id=740759716756)
- Some design decisions are outlined in an obsolete [RFC](https://docs.google.com/document/d/1Cc3772US-E73yQEMFn444OY9og9blKHpuP21sv9Gdxk/edit?usp=sharing)
- Project log is in [Notion](https://www.notion.so/AI-for-material-design-1f8f321d2ac54245a7af410d838929ae)
- Paper in [Overleaf](https://www.overleaf.com/project/61893015795e7b18e7979f53)