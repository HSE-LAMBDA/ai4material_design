const workflow_id = 'f7ef25e293394a1aa8ae2ba27b478e64';
const trial_prefix = '/ai4material_design/scripts/Rolos/workflows/run_experiment/combined_mixed_weighted_test/schnet/25-11-2022_16-52-31/';
const trial_groups = ['2a52dbe8', '71debf15'];

const workflow_id = '32de20e8eb174186a019f7c87b176bf2';
const trial_prefix = '/ai4material_design/scripts/Rolos/workflows/run_experiment/combined_mixed_weighted_test/megnet_pytorch/sparse/05-12-2022_19-50-53/';
const trial_groups = ['831cc496', 'd6b7ce45'];
const n_nodes = 3

const workflow_id = '0d0272e171d94f3485747b4b21505e21';
const trial_prefix = '/ai4material_design/scripts/Rolos/workflows/run_experiment/combined_mixed_weighted_test/megnet_pytorch/25-11-2022_11-38-18/';
const trial_groups = ['1baefba7'];
const n_nodes = 6

const workflow_id = 'e66fb0a25543458fb2bd9de46bed8a36';
const trial_prefix = '/ai4material_design/scripts/Rolos/workflows/run_experiment/combined_mixed_weighted_test/catboost/29-11-2022_13-16-01/';
const trial_groups = ['1b1af67c', '02e5eda9'];
const n_nodes = 6

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
	  "body": `{\"type\":\"environment\",\"name\":\"${trial_group}_${i}\",\"file_id\":\"${trial_prefix}/${trial_group}/node_${i}.sh\",\"environment_template_id\":\"7341a7991fc14514a5e087f700699665\",\"cpu_count\":8,\"ram\":32,\"gpu\":true,\"x\":${46+i*10},\"y\":${50+i*10}}`,
	  "method": "POST",
	  "mode": "cors",
	  "credentials": "include"
	});
	}
}
