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
const workflow_id = 'c3f94ae2585041c59e11cb85b48317a6';
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

// MegNet sparse
const target = 'formation_energy_per_site';
const workflow_id = 'c28e4dac763c43d4a5e3db19dacafd58';
const trial_prefix = `ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/${target}/megnet_pytorch/sparse/05-12-2022_19-50-53/`;
const trial_groups = ['d6b7ce45'];
const n_nodes = 3

const target = 'homo_lumo_gap_min';
const trial_prefix = `ai4material_design/scripts/Rolos/workflows/run_experiments/combined_mixed_weighted_test/${target}/megnet_pytorch/sparse/05-12-2022_19-50-53/`;
const trial_groups = ['831cc496'];
const n_nodes = 3

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

// VASP to csv/cif
const workflow_id = '387b10086f374c9a9e6a96be0db100ee';
const n_nodes = 8;

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
  "body": `{\"type\":\"environment\",\"name\":\"${i}_node\",\"file_id\":\"ai4material_design/scripts/Rolos/workflows/csv-cif/node_${i}.sh\",\"environment_template_id\":\"7341a7991fc14514a5e087f700699665\",\"cpu_count\":2,\"ram\":8,\"gpu\":false,\"x\":${46+Math.random()*1000},\"y\":${50+Math.random()*600}}`,
  "method": "POST",
  "mode": "cors",
  "credentials": "include"
});
}

// csv/cif to dataframe
const workflow_id = '4c17678444ce45e29ce1a0e0fefe7736';
const n_nodes = 8;

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
  "body": `{\"type\":\"environment\",\"name\":\"${i}_node\",\"file_id\":\"ai4material_design/scripts/Rolos/workflows/processed/node_${i}.sh\",\"environment_template_id\":\"7341a7991fc14514a5e087f700699665\",\"cpu_count\":2,\"ram\":8,\"gpu\":false,\"x\":${46+Math.random()*1000},\"y\":${50+Math.random()*600}}`,
  "method": "POST",
  "mode": "cors",
  "credentials": "include"
});
}

// matminer
const workflow_id = 'bdc93234c26a4ce891df1b10a762c3c4';
const n_nodes = 8;
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
  "body": `{\"type\":\"environment\",\"name\":\"${i}_node\",\"file_id\":\"ai4material_design/scripts/Rolos/workflows/matminer/node_${i}.sh\",\"environment_template_id\":\"7341a7991fc14514a5e087f700699665\",\"cpu_count\":24,\"ram\":48,\"gpu\":false,\"x\":${46+Math.random()*1000},\"y\":${50+Math.random()*600}}`,
  "method": "POST",
  "mode": "cors",
  "credentials": "include"
});
}

// Training models for inference
const workflow_id = "daa21a5b24a74109bddc8398a20068e5";

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
	"body": `{\"type\":\"environment\",\"name\":\"Train both models\",\"file_id\":\"ai4material_design/scripts/Rolos/workflows/final_training/train_both_targets.sh\",\"environment_template_id\":\"7341a7991fc14514a5e087f700699665\",\"cpu_count\":8,\"ram\":16,\"gpu\":true,\"x\":${46+Math.random()*1000},\"y\":${50+Math.random()*600}}`,
	"method": "POST",
	"mode": "cors",
	"credentials": "include"
  });