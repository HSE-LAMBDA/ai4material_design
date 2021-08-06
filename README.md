# AI for material design
## Pure-defect GNN
* Step 1: [data preparation](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/Defect%20representation.ipynb).
* Step 2: Trainining with `python megnet_graphs_train.py`.
* Step 3: [Plot and predict](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/megnet-05-data-defect-only-plot-predict.ipynb)
* Step 4: [Summary analysis of experiments](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/Summary%20analysis.ipynb)

The models are saved into [MEGNet-defect-only.dvc](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/models/MEGNet-defect-only.dvc), predictions and plots to [predicted_dichalcogenides_innopolis_202105_v3](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/datasets/predicted_dichalcogenides_innopolis_202105_v3).

## Defect screening with matminer
* [Defect generation](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/defects_generation/generation.ipynb)
* [Defect featurization](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/structure_featurization.py), featurize - simple structure features (~2k), featurize_expanded - wide set of features (~10k)

## Potential fitting on relaxation trajectories
* Graph neural network [training](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/GNN-traj.ipynb), [relaxation](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/GNN-relaxation.ipynb)

## Old MoS2 experiments
* CGCNN: [Paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.145301), [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/cgcnn.ipynb).
* Behler-Parrinello: [paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401), [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/BP-wide-tuned-no-force.ipynb) Nikita has modified the potential to provide more information about the atom species, and made the descriptor parameters learnable, to no avail.
* Graph neural network: [paper](https://www.nature.com/articles/s41567-020-0842-8), [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/GraphNN-wide.ipynb)
* Set transformer: [paper](http://proceedings.mlr.press/v97/lee19d.html), [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/SetTransformer.ipynb)
* Expert-created potential: [paper](https://aip.scitation.org/doi/10.1063/1.5007842), [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/KIM%20potential.ipynb)

The experiments use the code from [Jax MD example](https://colab.research.google.com/github/google/jax-md/blob/master/notebooks/neural_networks.ipynb)
## Getting the data from DVC
Installing dvc: `pip install 'dvc[azure]'`

Getting the data: `dvc fetch && dvc checkout`

[DVC documentation](https://dvc.org/doc)
    
