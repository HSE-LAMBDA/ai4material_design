# Generating the workflow scripts for the Constructor Research Platform
The scripts and workflows are already on the platform. This section is for reference only.
## Data preprocessing
1. Generate the platform scripts from DVC
```bash
cd ai4material_design
./scripts/Rolos/generate_workflow_scrtipts_from_dvc.sh 8
```
2. Create the workflows
 - Create the workflows manually using the UI
 - Put your workflow and project ids to [`../scripts/Rolos/create_workflows.js`](../scripts/Rolos/create_workflows.js)
 - Log in to the platform, open the browser console, paste the relevant parts from [`../scripts/Rolos/create_workflows.js`](../scripts/Rolos/create_workflows.js). You need to do it for each workflow.
## Computational experiments
1. Generate the scripts:
```bash
cd ai4material_design/scripts/Rolos
xargs -a stability_trials.txt -L1 ./generate_experiments_workflow.sh 
```
2. Create the workflows
 -  Create the workflows manually using the UI
 - Put your workflow and project ids to [`../scripts/Rolos/create_workflows.js`](../scripts/Rolos/create_workflows.js)
 -  Log in to the platform, open the browser console, paste the relevant parts from [`../scripts/Rolos/create_workflows.js`](../scripts/Rolos/create_workflows.js). You need to do it for each workflow.
 ## README
 5. Add the global README, update the relative links.
```bash
cp ai4material_design/docs/CONSTRUCTOR.md README.md
sed -i -E 's|(\[[^]]+]\()([^/h)][^)]+\))|\1ai4material_design/docs/\2|g' README.md
```