import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure import *
from matminer.featurizers.composition import *
import pymatgen

from tqdm.auto import tqdm
tqdm.pandas()

import sys
sys.path.append('../ai4material_design')

from data import get_dichalcogenides_innopolis


compute_all = False


def featurize_structure(structure: pymatgen.core.structure.Structure) -> dict:
    """Computes features for one pymatgen structure"""
    feature_calculator = MultipleFeaturizer([
        DensityFeatures(),
        OrbitalFieldMatrix(),
        StructuralHeterogeneity(),
        MaximumPackingEfficiency(),
        XRDPowderPattern(), 
        JarvisCFID(),
        StructuralComplexity()
    ])
    features = feature_calculator.featurize(structure)
    features = dict(zip(feature_calculator.feature_labels(), features))
    return features


def main():
    print("Loading defects:")
    defects, descriptors = get_dichalcogenides_innopolis('datasets/dichalcogenides_innopolis_202105/')
    if not compute_all:
        defects = defects[:5]
    print("Computing features:")
    features = defects['initial_structure'].progress_apply(featurize_structure)
    features_df = pd.DataFrame(list(features), index = features.index)
    features_df.to_csv('datasets/paper_experiments_catboost/features/computed_features.csv')
    
    
if __name__ == "__main__":
    main()