import pymatgen
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.structure.order import (
    DensityFeatures,
    MaximumPackingEfficiency,
    StructuralComplexity)
from matminer.featurizers.structure.matrix import OrbitalFieldMatrix
from matminer.featurizers.structure.misc import XRDPowderPattern
from matminer.featurizers.structure.composite import JarvisCFID


def featurize(structure: pymatgen.core.structure.Structure) -> dict:
    """Computes features for one pymatgen structure"""
    feature_calculator = MultipleFeaturizer([DensityFeatures(),
                                             OrbitalFieldMatrix(),
                                             MaximumPackingEfficiency(),
                                             XRDPowderPattern(), 
                                             JarvisCFID(), 
                                             StructuralComplexity()])
    features = feature_calculator.featurize(structure)
    features = dict(zip(feature_calculator.feature_labels(), features))
    return features


def rdf_transform(features):
    """Parses results of RDF featurization"""
    rdf = features['radial distribution function']
    rdf_keys = [f"rdf{i}" for i in range(len(rdf['distances']))]
    rdf_values = rdf['distribution']
    
    del features['radial distribution function']
    features = {**features, **dict(zip(rdf_keys, rdf_values))}
    return features


def erdf_transform(features):
    """Parses results of ERDF featurization"""
    erdf = features['electronic radial distribution function']
    erdf_keys = [f"erdf{i}" for i in range(len(erdf['distances']))]
    erdf_values = erdf['distribution']
    
    del features['electronic radial distribution function']
    features = {**features, **dict(zip(erdf_keys, erdf_values))}
    return features       


def featurize_expanded(structure: pymatgen.core.structure.Structure, 
                               guess_oxidation: bool=True):
    """Computes expanded set of features for one pymatgen structure"""
    if guess_oxidation:
        structure.add_oxidation_state_by_element(
            {"Mo":4, "W":4, "S":-2, "Se":-2, "O":-2})
    prdf = PartialRadialDistributionFunction(
        include_elems = ("W", "Mo", "Se", "S", "O")).fit([structure])
    coulomb = CoulombMatrix().fit([structure])
    coulomb._max_eigs = max(coulomb._max_eigs, 110)
    sinecoulomb = SineCoulombMatrix().fit([structure])
    sinecoulomb._max_eigs = max(sinecoulomb._max_eigs, 110)
    bondFractions = BondFractions().fit([structure])
    
    composition_feature_calculator = MultipleFeaturizer([
        ElementProperty.from_preset('matminer'),
        Meredig(), 
        OxidationStates(),
        AtomicOrbitals(),
        BandCenter(),
        ElectronegativityDiff(),
        ElectronAffinity(), 
        Stoichiometry(), 
        ValenceOrbital(),
        IonProperty(),
        ElementFraction(),
        TMetalFraction(),
        Miedema(),
        YangSolidSolution(),
        AtomicPackingEfficiency()
    ])
        
    structure_feature_calculator = MultipleFeaturizer([
        DensityFeatures(desired_features=None),  # =None means all accessible features
        EwaldEnergy(),
        GlobalSymmetryFeatures(desired_features=None),
        RadialDistributionFunction(cutoff=80.0, bin_size=0.1),
        ElectronicRadialDistributionFunction(cutoff=80.0, dr=0.05),
        coulomb,
        sinecoulomb,
        OrbitalFieldMatrix(period_tag=True, flatten=True),
        SiteStatsFingerprint.from_preset('CrystalNNFingerprint_cn'),
        SiteStatsFingerprint.from_preset('OPSiteFingerprint'),
        SiteStatsFingerprint.from_preset('LocalPropertyDifference_ward-prb-2017'),
        bondFractions,
        StructuralHeterogeneity(),
        MaximumPackingEfficiency(),
        ChemicalOrdering(),
        StructureComposition(composition_feature_calculator),
        XRDPowderPattern(), 
        JarvisCFID(), 
        GlobalInstabilityIndex(),
        StructuralComplexity(),
        prdf,  ## should be the last one as it deletes oxidation states (?!)
    ])
    features = structure_feature_calculator.featurize(structure)
    features = dict(zip(structure_feature_calculator.feature_labels(), features))
    features = rdf_transform(features)
    features = erdf_transform(features)
    features['compound possible'] = bool(features['compound possible'])
    return features