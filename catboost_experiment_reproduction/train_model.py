import pandas as pd
import os
import itertools
from catboost import CatBoostRegressor, Pool
from tqdm.auto import tqdm

MODEL_PATH_ROOT = os.path.join("datasets", "paper_experiments_catboost", "models")


class Experiment():
    def __init__(self,
                 data_path: str,
                 folds_path: str,
                 features_path: str,
                 total_folds: int,
                 test_fold: list,
                 target: str,
                 epochs: int = 1000,
                 learning_rate: float = 0.1,
                 supercell_replication = None):
        self.name = (f"fold_{test_fold}"
                     f"_{epochs}"
                     ".cbm")
        self.target = target
        self.epochs = epochs
        self.model_path = os.path.join(MODEL_PATH_ROOT, self.target, self.name)
        self.learning_rate = learning_rate
        self.data_path = data_path
        self.folds_path = folds_path
        self.features_path = features_path
        self.total_folds = total_folds
        self.test_fold = test_fold
        
    def run(self, gpu=None):
        self.__load_data()
        self.__train_model()
        
    def __load_data(self,):
        data = pd.read_csv(self.data_path).set_index("_id")
        folds = pd.read_csv(self.folds_path, squeeze=True, index_col="_id")
        features = pd.read_csv(self.features_path).set_index("_id")
        train_folds = set(range(self.total_folds)) - set((self.test_fold,))
        train_ids = folds[folds.isin(train_folds)]
        train = data.reindex(index=train_ids.index)
        train_target = train[self.target]

        test_ids = folds[folds == self.test_fold]
        test = data.reindex(index=test_ids.index)
        test_target = test[self.target]
        
        self.train_features = features.reindex(index=train_ids.index)
        self.test_features = features.reindex(index=test_ids.index)
        self.train_target = train_target
        self.test_target = test_target
        
    def __train_model(self,):
        model = CatBoostRegressor(iterations=self.epochs, loss_function="MAE",
                                  verbose=False, random_seed=0)
        model.fit(self.train_features, self.train_target)
        model.save_model(self.model_path, format='cbm')
        
        
# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
        
        
def generate_paper_experiments():
    params = {
        "learning_rate": (0.2,),
        "target": ("energy_per_atom",),# "homo", "band_gap"),
        "epochs": (1000,),
        "total_folds": (8,),
        "test_fold": range(8),
        "data_path": ("datasets/paper_experiments_catboost/targets/defects.csv",),
        "folds_path": ("datasets/paper_experiments_catboost/inputs/full.csv",),
        "features_path": ("datasets/paper_experiments_catboost/features/features.csv", )
    }
    return [Experiment(**params) for params in product_dict(**params)]

        
def main():
    experiments = generate_paper_experiments()
    print("Running experiments:")
    for experiment in tqdm(experiments):
        experiment.run()
        
 
if __name__ == '__main__':
    main()