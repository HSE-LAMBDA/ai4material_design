import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
WORKFLOWS_DIR = os.path.join(SCRIPT_DIR, "workflows")

def dump_dvc(workflow, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    commands = subprocess.run(
        ["dvc", "repro", "-s", "-f", "--dry", workflow],
        capture_output=True,
        text=True
    ).stdout
    with open(os.path.join(output_dir, "commands.txt"), "w") as file:
        file.write(commands)

if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    print("Usage: python script.py number_of_nodes", file=sys.stderr)
    sys.exit(1)

number_of_nodes = int(sys.argv[1])
print(f"Generating scripts for {number_of_nodes} nodes")

os.chdir(SCRIPT_DIR)
os.chdir("../../")
params_yaml_bak = os.path.join(SCRIPT_DIR, "../../params.yaml.bak")
params_yaml = os.path.join(SCRIPT_DIR, "../../params.yaml")
subprocess.run(["cp", params_yaml, params_yaml_bak])

subprocess.run(["cp", "../../params-rolos-workflow.yaml", params_yaml])
print("Copied params-rolos-workflow.yaml to params.yaml")

dump_dvc("csv-cif-low-density-8x8-Innopolis-v1", os.path.join(WORKFLOWS_DIR, "low-density-index"))
dump_dvc("csv-cif-no-spin-500-data csv-cif-spin-500-data csv-cif-low-density-8x8", os.path.join(WORKFLOWS_DIR, "csv-cif"))
dump_dvc("processed-low-density processed-high-density", os.path.join(WORKFLOWS_DIR, "processed"))
dump_dvc("matminer@high_density_defects/BP_spin_500 matminer@high_density_defects/GaSe_spin_500 matminer@high_density_defects/hBN_spin_500 matminer@high_density_defects/InSe_spin_500 matminer@high_density_defects/MoS2_500 matminer@high_density_defects/WSe2_500 matminer@low_density_defects/MoS2 matminer@low_density_defects/WSe2", os.path.join(WORKFLOWS_DIR, "matminer"))

for workflow in ["low-density-index", "csv-cif", "processed", "matminer"]:
    for node in range(1, number_of_nodes + 1):
        print(f"Generating script for {workflow} node {node}")
        filename = os.path.join(WORKFLOWS_DIR, workflow, f"node_{node}.sh")
        with open(filename, "w") as file:
            file.write("#!/bin/bash\n")
            file.write("cd ai4material_design\n")
            file.write("if [ ! -f scripts/Rolos/dry-run ]; then\n")
            if workflow == "matminer":
                file.write('pip install "numpy<1.24.0"\n')
            file.write(
                subprocess.run(
                    ["awk", f'(NR - 1) % {number_of_nodes} == {node - 1}', os.path.join(WORKFLOWS_DIR, workflow, "commands.txt")],
                    capture_output=True,
                    text=True
                ).stdout
            )
            file.write("fi\n")

subprocess.run(["mv", params_yaml_bak, params_yaml])
print("Restored params.yaml")
