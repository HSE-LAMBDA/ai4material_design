const workflow_id = 'f7ef25e293394a1aa8ae2ba27b478e64'
const trial_prefix = '/ai4material_design/scripts/Rolos/workflows/run_experiment/combined_mixed_weighted_test/schnet/25-11-2022_16-52-31/'
for (trial_group of ['2a52dbe8', '71debf15']) {
	for (let i = 0; i < 13; i++) {
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
	  "body": `{\"type\":\"environment\",\"name\":\"${trial_group}_${i}\",\"file_id\":\"${trial_prefix}/${trial_group}/node_${i}.sh\",\"environment_template_id\":\"7341a7991fc14514a5e087f700699665\",\"cpu_count\":6,\"ram\":24,\"gpu\":true,\"x\":${46+i*10},\"y\":${50+i*10}}`,
	  "method": "POST",
	  "mode": "cors",
	  "credentials": "include"
	});
	}
}
