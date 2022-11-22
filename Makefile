
get_gemnet_tables:
	python scripts/summary_table.py \
		--experiments high_density/GaSe_500    \
					high_density/InSe_500    \
					high_density/MoS2_500    \
					high_density/WSe2_500    \
		--trials gemnet-full \
		--targets homo lumo formation_energy_per_site \
		--separate-by trial

	python scripts/summary_table.py \
		--experiments high_density/BP_spin_500 \
					high_density/hBN_spin_500  \
		--trials gemnet-full \
		--targets formation_energy_per_site \
					band_gap_majority \
					band_gap_minority \
					homo_majority \
					homo_minority \
					lumo_majority \
					lumo_minority \
					total_mag \
		--separate-by trial

get_gemnet_tables_combined:
	python scripts/summary_table.py \
		--experiments \
					high_density/combined    \
					high_density/combined-2    \
		--trials gemnet-full \
		--targets homo_majority lumo_majority formation_energy_per_site \
		--separate-by trial

get_gemnet_plots_combined:
	python scripts/plot.py \
		--experiments high_density/combined high_density/combined-2 --trials gemnet-full

get_megnet_upsampling_plots:
	python scripts/plot.py --experiments combined_mixed_upsampling_minority_test \
	 --trials megnet_pytorch-sparse-z-were \
	 --strategy train_test

get_gemnet_plots:
	python scripts/plot.py \
		--experiments high_density/BP_spin_500 high_density/GaSe_500 high_density/hBN_spin_500 high_density/InSe_500 high_density/MoS2_500 high_density/WSe2_500  --trials gemnet-full
	