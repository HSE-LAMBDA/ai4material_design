# AI for material design
## Potential fitting on relaxation trajectories
* Graph neural network [training](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/GNN-traj.ipynb), [relaxation](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/GNN-relaxation.ipynb)

## Defect generation
* [Example of usage](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/defects_generation/generation.ipynb)
## Old MoS2 experiments
* Behler-Parrinello: [paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401), [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/BP-wide-tuned-no-force.ipynb) Nikita has modified the potential to provide more information about the atom species, and made the descriptor parameters learnable, to no avail.
* Graph neural network: [paper](https://www.nature.com/articles/s41567-020-0842-8), [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/GraphNN-wide.ipynb)
* Set transformer: [paper](http://proceedings.mlr.press/v97/lee19d.html), [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/SetTransformer.ipynb)
* Expert-created potential: [paper](https://aip.scitation.org/doi/10.1063/1.5007842), [code](https://github.com/HSE-LAMBDA/ai4material_design/blob/main/KIM%20potential.ipynb)

The experiments use the code from [Jax MD example](https://colab.research.google.com/github/google/jax-md/blob/master/notebooks/neural_networks.ipynb)
## Getting the data from DVC
Installing dvc: `pip install 'dvc[azure]'`

Getting the data: `dvc fetch && dvc checkout`

[DVC documentation](https://dvc.org/doc)
