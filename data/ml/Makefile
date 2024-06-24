.PHONY: deploy and run

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(lastword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

deploy-image-data: ## "Deploying data image"
	@sh bin/deploy-image.sh --target data

deploy-image-model: ## "Deploying model image"
	@sh bin/deploy-image.sh --target model

deploy-image-inference: ## "Deploying inference image"
	@sh bin/deploy-image.sh --target inference

deploy-code-data: ## Deploy EMR data code.
	./bin/deploy-code.sh --target data

deploy-code-model: ## Deploy Sagemaker model code.
	./bin/deploy-code.sh --target model

deploy-code-inference: ## Deploy Sagemaker inference code.
	./bin/deploy-code.sh --target inference

py-run-data: ## Deploy code and Run data job.
	./bin/deploy-code.sh --target data && ./bin/run-data.sh

py-run-model: ## Deploy code and Run Sagemaker training job.
	./bin/deploy-code.sh --target model && ./bin/run-model.sh

py-run-inference: ## Deploy code and Run Sagemaker inference job.
	./bin/deploy-code.sh --target inference && ./bin/run-inference.sh

run-data: ## Run data job.
	./bin/run-data.sh

run-model: ## Run Sagemaker training job.
	./bin/run-model.sh

run-inference: ## Run Sagemaker inference job.
	./bin/run-inference.sh
