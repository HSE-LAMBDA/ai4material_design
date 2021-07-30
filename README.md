# AI for material design
## Pure-defect GNN
* Step 1: [data preparation](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/Defect%20representation.ipynb).
* Step 2: trainining with `megnet_graphs_train.py`.
```
python megnet_graphs_train.py --train datasets/train_defects.pickle.gzip --target homo --is-intensive True --experiment-name clean_full
python megnet_graphs_train.py --train datasets/train_defects.pickle.gzip --target formation_energy --is-intensive False --experiment-name clean_full
python megnet_graphs_train.py --train datasets/train_defects_vac_only.pickle.gzip --target homo --is-intensive True --experiment-name clean_vac_only
python megnet_graphs_train.py --train datasets/train_defects_vac_only.pickle.gzip --target formation_energy --is-intensive False --experiment-name clean_vac_only
```
* [Plot and predict](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/megnet-05-data-defect-only-plot-predict.ipynb)

The models are saved into `https://github.com/HSE-LAMBDA/ai4material_design/blob/main/models/MEGNet-defect-only.dvc` and predictions to `https://github.com/HSE-LAMBDA/ai4material_design/blob/main/datasets/predicted_dichalcogenides_innopolis_202105_v2`.

## Potential fitting on relaxation trajectories
* Graph neural network [training](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/GNN-traj.ipynb), [relaxation](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/GNN-relaxation.ipynb)

## Defect screening
* [Defect generation](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/defects_generation/generation.ipynb)
* [Defect featurization](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/structure_featurization.py), featurize - simple structure features (~2k), featurize_expanded - wide set of features (~10k)
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
    
