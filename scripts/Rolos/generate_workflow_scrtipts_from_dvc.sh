#!/bin/bash
SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
WORKFLOWS_DIR=$SCRIPT_DIR/workflows
re='^[0-9]+$'
if ! [[ $1 =~ $re ]] ; then
   echo "Usage: $0 number_of_nodes" >&2; exit 1
fi
echo "Generating sctipts for $1 nodes"
dump_dvc () {
    mkdir -p $WORKFLOWS_DIR/$2
    dvc repro -s -f --dry $1 | grep "^> " | cut -c 3- > $WORKFLOWS_DIR/$2/commands.txt
}

cp $SCRIPT_DIR/../../params.yaml params.yaml.bak
cp $SCRIPT_DIR/../../params-rolos-workflow.yaml $SCRIPT_DIR/../../params.yaml
echo "Copied params-rolos-workflow.yaml to params.yaml"

# We separate the dvc pipeilne into chunks that can be run in parallel
dump_dvc csv-cif-low-density-8x8-Innopolis-v1 low-density-index
dump_dvc "csv-cif-no-spin-500-data csv-cif-spin-500-data csv-cif-low-density-8x8" csv-cif 
dump_dvc "processed-low-density processed-high-density" processed
dump_dvc matminer matminer

for workflow in low-density-index csv-cif processed matminer; do
    for node in  $(seq 1 $1); do
        echo "Generating script for $workflow node $node"
        filename=$WORKFLOWS_DIR/$workflow/"node_$node".sh
        echo "#!/bin/bash" > $filename
        echo "cd ai4material_design" >> $filename
        awk -v NUM=$1 -v NODE=$node '(NR - 1) % NUM == NODE - 1' $WORKFLOWS_DIR/$workflow/commands.txt >> $filename
    done
done

mv $SCRIPT_DIR/../../params.yaml.bak $SCRIPT_DIR/../../params.yaml
echo "Restored params.yaml"