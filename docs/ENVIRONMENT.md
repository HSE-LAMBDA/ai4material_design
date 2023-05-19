# Setting up the environment
1. [Install Poetry](https://python-poetry.org/docs/#installation)
2. Next steps depend on your setup
   - If you don't want to use vritualenv, for example to use system `torch`, run
   ```
   poetry config virtualenvs.create false --local
   ```
   - If you want to use virtualenv, run
   ```
   poetry shell
   ```
3. Install ordinary dependencies
```
poetry install
```
If it fails, try removing `poetry.lock`. We support multiple Python versions, so even if the lock file doesn't work, you still might be able install the packages.

4. [Install pytorch](https://pytorch.org/) according to your CUDA/virtualenv/conda situatoin
5. [Install pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) according to your CUDA/virtualenv/conda situation
6. [Log in to WanDB](https://docs.wandb.ai/ref/cli/wandb-login), or set `WANDB_MODE=disabled`