.PHONY: start test test-real

start: ## Run the inference service on :5001 (set AI_SERVICE_TOKEN to require the header)
	venv/bin/uvicorn main:app --host 0.0.0.0 --port 5001

test: ## Fast tests with a fake predictor
	venv/bin/python -m pytest -q

test-real: ## Also exercise the real model on the sample images (~10 s)
	AI_REAL_MODEL=1 venv/bin/python -m pytest -q -k real_model
