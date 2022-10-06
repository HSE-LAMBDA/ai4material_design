get_predictions:
	cd processed
	dvc pull -r ai4materialdesign dichalcogenides_x1s6_202109_MoS2.dvc
	cd ../experiments
	dvc pull -r ai4materialdesign MoS2-plain-cv.dvc
	cd ../predictions
	dvc pull -r ai4materialdesign *dvc
