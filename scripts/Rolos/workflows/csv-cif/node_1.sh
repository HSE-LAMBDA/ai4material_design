#!/bin/bash
cd ai4material_design
if [ ! -f scripts/Rolos/dry-run ]; then
ratarmount -o modules=subdir,subdir=wse2_8x8_5933 datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110/wse2_8x8_5933.tar.gz /output/datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110/WSe2_8x8_5933/ && python scripts/vasp_to_csv_cif.py --input-vasp "/output/datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110/WSe2_8x8_5933" --input-structures-list datasets/csv_cif/low_density_defects_Innopolis-v1/WSe2 --input-structures-csv-cif datasets/csv_cif/low_density_defects_Innopolis-v1/WSe2 --output-csv-cif "/output/datasets/csv_cif/low_density_defects/WSe2" --pristine-folder datasets/others/pristine_high_density --vasprun-glob-prefix poscar_??- && sleep 5 && ratarmount -u /output/datasets/raw_vasp/dichalcogenides8x8_vasp_nus_202110/WSe2_8x8_5933/
fi
