// CatBoost
const workflow_id = '6ab90a79599e42cc9843902238bd6305';
const target = 'formation_energy_per_site';
const trial_prefix = `ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/${target}/catboost/29-11-2022_13-16-01/`;
const trial_groups = ['02e5eda9'];
const n_nodes = 6;

const workflow_id = '6ab90a79599e42cc9843902238bd6305';
const target = 'homo_lumo_gap_min';
const trial_prefix = `ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/${target}/catboost/29-11-2022_13-16-01/`;
const trial_groups = ['1b1af67c'];
const n_nodes = 6;

// SchNet
const workflow_id = '3df5b3925c5c4483a1722c7c55862e03';
const n_nodes = 12;

const target = 'formation_energy_per_site';
const trial_prefix = 'ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/formation_energy_per_site/schnet/25-11-2022_16-52-31/';
const trial_groups = ['71debf15'];

const target = 'homo_lumo_gap_min';
const trial_prefix = 'ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/homo_lumo_gap_min/schnet/25-11-2022_16-52-31/';
const trial_groups = ['2a52dbe8'];

// GemNet
const workflow_id = '37faed757811498c90536129468c390e';
const n_nodes = 12;

const target = 'formation_energy_per_site';
const trial_prefix = 'ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/formation_energy_per_site/gemnet/16-11-2022_20-05-04/';
const trial_groups = ['b5723f85'];

const target = 'homo_lumo_gap_min';
const trial_prefix = 'ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/homo_lumo_gap_min/gemnet/16-11-2022_20-05-04/';
const trial_groups = ['c366c47e']

// MegNet Full
const target = 'formation_energy_per_site';
const workflow_id = 'da6fa82ca6d04c0b90700c6b1c8d242c';
const trial_prefix = `ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/${target}/megnet_pytorch/25-11-2022_11-38-18/`;
const trial_groups = ['1baefba7'];
const n_nodes = 12

const target = 'homo_lumo_gap_min';
const workflow_id = 'da6fa82ca6d04c0b90700c6b1c8d242c';
const trial_prefix = `ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/${target}/megnet_pytorch/25-11-2022_11-38-18/`;
const trial_groups = ['1baefba7'];
const n_nodes = 12

for (trial_group of trial_groups) {
	for (let i = 1; i <= n_nodes; i++) {
	    fetch(`https://my.rolos.com/api/v1/workflows/${workflow_id}/nodes`, {
	  "headers": {
	    "accept": "*/*",
	    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
	    "content-type": "application/json",
	    "sec-ch-ua": "\"Not A(Brand\";v=\"24\", \"Chromium\";v=\"110\"",
	    "sec-ch-ua-mobile": "?0",
	    "sec-ch-ua-platform": "\"Linux\"",
	    "sec-fetch-dest": "empty",
	    "sec-fetch-mode": "cors",
	    "sec-fetch-site": "same-origin"
	  },
	  "referrer": `https://my.rolos.com/projects/79a29e5d84da4e5680ed6d8c9f933748/workflows/${workflow_id}`,
	  "referrerPolicy": "strict-origin-when-cross-origin",
	  "body": `{\"type\":\"environment\",\"name\":\"${target}_${trial_group}_${i}\",\"file_id\":\"${trial_prefix}/${trial_group}/node_${i}.sh\",\"environment_template_id\":\"7341a7991fc14514a5e087f700699665\",\"cpu_count\":8,\"ram\":32,\"gpu\":true,\"x\":${46+Math.random()*1000},\"y\":${50+Math.random()*600}}`,
	  "method": "POST",
	  "mode": "cors",
	  "credentials": "include"
	});
	}
}