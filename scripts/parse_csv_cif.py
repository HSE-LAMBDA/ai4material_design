import argparse
from operator import methodcaller
from pathlib import Path
import logging
import pandas as pd
from pymatgen.core.periodic_table import Element, DummySpecies

import sys
sys.path.append('.')
from MEGNetSparse.dense2sparse import convert_to_sparse_representation, add_was
from MEGNetSparse.eos import EOS
from ai4mat.data.data import (
    get_dichalcogenides_innopolis,
    StorageResolver,
    Columns,
    get_unit_cell
)


SINGLE_ENENRGY_COLUMN = "chemical_potential"


def energy_correction(structure, single_atom_energies):
    correction = 0
    for site in structure.sites:
        if isinstance(site.specie, DummySpecies):
            correction += single_atom_energies.loc[Element.from_Z(site.properties['was']), SINGLE_ENENRGY_COLUMN]
        else:
            correction -= single_atom_energies.loc[site.specie, SINGLE_ENENRGY_COLUMN]
            correction += single_atom_energies.loc[Element.from_Z(site.properties['was']), SINGLE_ENENRGY_COLUMN]
    return correction


def parse_csv_cif(input_folder, args, dataset_name):
    structures, defects = get_dichalcogenides_innopolis(input_folder)
    if (args.fill_missing_band_properties and
            "homo_lumo_gap" not in structures.columns and
            "homo" in structures.columns and
            "lumo" in structures.columns):
        structures["homo_lumo_gap"] = structures["lumo"] - structures["homo"]

    materials = defects.base.unique()
    unit_cells = get_unit_cell(input_folder, materials)
    if not args.skip_eos:
        for material in materials:
            unit_cells[material] = EOS().get_augmented_struct(unit_cells[material])

    data_path = Path(input_folder)
    initial_structure_properties = pd.read_csv(
        data_path.joinpath("initial_structures.csv"),
        converters={"cell_size": lambda x: tuple(eval(x))},
        index_col=["base", "cell_size"])
    single_atom_energies = pd.read_csv(data_path.joinpath("elements.csv"),
                                       index_col="element",
                                       converters={"element": Element})

    COLUMNS = Columns()

    # TODO(kazeevn) this all is very ugly
    def get_defecs_from_row(row):
        defect_description = defects.loc[row[COLUMNS["structure"]["descriptor_id"]]]
        unit_cell = unit_cells[defect_description.base]
        initial_energy = initial_structure_properties.at[
            (defect_description.base, defect_description.cell), "energy"]
        defect_structure = convert_to_sparse_representation(
            row.initial_structure,
            unit_cell,
            defect_description.cell,
            skip_eos=True,
            copy_unit_cell_properties=True
        )
        structure_with_was = add_was(
            row.initial_structure,
            unit_cell,
            defect_description.cell,
        )
        formation_energy_part = energy_correction(defect_structure, single_atom_energies)
        return defect_structure, formation_energy_part + row.energy - initial_energy, structure_with_was

    defect_properties = structures.apply(get_defecs_from_row,
                                         axis=1,
                                         result_type="expand")

    structures = structures.drop(COLUMNS["structure"]["unrelaxed"], axis=1)
    defect_properties.columns = [
        COLUMNS["structure"]["sparse_unrelaxed"],
        "formation_energy",
        COLUMNS["structure"]["unrelaxed"]
    ]
    structures = structures.join(defect_properties)
    structures["formation_energy_per_site"] = structures[
                                                  "formation_energy"] / structures[
                                                  COLUMNS["structure"]["sparse_unrelaxed"]].apply(len)
    structures["energy_per_atom"] = structures["energy"] / structures[COLUMNS["structure"]["unrelaxed"]].apply(len)

    assert structures.apply(lambda row: len(row[COLUMNS["structure"]["sparse_unrelaxed"]]) == len(
        defects.loc[row[COLUMNS["structure"]["descriptor_id"]], "defects"]), axis=1).all()

    if args.normalize_homo_lumo:
        for kind in (None, "majority", "minority"):
            for property in ("homo", "lumo"):
                if kind is None:
                    column = property
                else:
                    column = f"{property}_{kind}"
                if column not in structures.columns:
                    continue
                defects_per_structure = defects.loc[structures[COLUMNS["structure"]["descriptor_id"]]]
                defects_key = (defects_per_structure.base.unique(), defects_per_structure.cell.unique())
                if len(defects_key[0]) != 1 or len(defects_key[1]) != 1:
                    raise NotImplementedError("Handling different pristine materials in same dataset not implemented")
                defects_key = (defects_key[0][0], defects_key[1][0])
                normalization_constant = initial_structure_properties.at[defects_key, "E_VBM"] - \
                                         initial_structure_properties.at[defects_key, "E_1"]
                structures[f"normalized_{column}"] = \
                    structures[column] - structures["_".join(filter(None, ("E_1", kind)))] - normalization_constant

    for property in ("homo_lumo_gap", "homo", "lumo", "normalized_homo", "normalized_lumo"):
        for kind in ("min", "max"):
            column = f"{property}_{kind}"
            if column in structures.columns:
                raise ValueError(f"Column {column} already exists, it's not supposed to at this stage")
            source_columns = [f"{property}_majority", f"{property}_minority"]
            if not frozenset(source_columns).issubset(structures.columns):
                logging.info("Skipped filling %s, as %s_{majority,minority} are not available", column, property)
                continue
            structures[column] = methodcaller(kind, axis=1)(structures.loc[:, source_columns])

    if args.fill_missing_band_properties:
        for kind in ("majority", "minority", "max", "min"):
            for property in ("homo_lumo_gap", "homo", "lumo", "normalized_homo", "normalized_lumo", "E_1"):
                spin_column = f"{property}_{kind}"
                if spin_column not in structures.columns:
                    if property in structures.columns:
                        structures[spin_column] = structures[property]
                        logging.info("Filling {}", spin_column)
                    else:
                        logging.warning(r"%s is missing in data, can't fill %s", property, spin_column)
        if "total_mag" not in structures.columns:
            structures["total_mag"] = 0.
            logging.info("Setting total_mag = 0")

    save_dir = StorageResolver(root_folder=args.output_folder)["processed"].joinpath(dataset_name)
    save_dir.mkdir(exist_ok=True, parents=True)
    structures.to_pickle(
        save_dir.joinpath("data.pickle.gz"))
    structures.drop(columns=[
        COLUMNS["structure"]["unrelaxed"],
        COLUMNS["structure"]["sparse_unrelaxed"]]
    ).to_csv(save_dir.joinpath("targets.csv.gz"),
             index_label=COLUMNS["structure"]["id"])


def main():
    parser = argparse.ArgumentParser("Parses csv/cif into pickle and targets.csv.gz")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-folder", type=str)
    group.add_argument("--input-name", type=str)
    parser.add_argument("--fill-missing-band-properties", action="store_true")
    parser.add_argument("--normalize-homo-lumo", action="store_true")
    parser.add_argument("--skip-eos", action="store_true",
                        help="Don't add EOS indices")
    parser.add_argument("--output-folder", type=Path,
                        help="Path where to write the output. "
                             "The usual directory structure 'datasets/processed/<dataset_name>'"
                             "will be created.")
    args = parser.parse_args()

    storage_resolver = StorageResolver()
    if args.input_folder:
        dataset_name = Path(args.input_folder).name
        input_folder = args.input_folder
    else:
        dataset_name = args.input_name
        input_folder = storage_resolver["csv_cif"].joinpath(dataset_name)

    parse_csv_cif(input_folder, args, dataset_name)


if __name__ == "__main__":
    main()
