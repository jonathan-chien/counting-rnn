run-demo:
	python scripts/run_train.py

update-environment-yaml:
	conda env export > environment.yaml

	