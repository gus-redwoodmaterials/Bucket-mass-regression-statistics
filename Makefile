export UV_INDEX?=$(shell pip config get global.index-url)

.PHONY: help
help:  ## Show available options
	@echo
	@echo "\033[1mUsage\033[0m: make <COMMAND>\n"
	@echo "\033[1mCommands\033[0m:\n"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: install  ## Build docker image for each service
	docker compose --profile=dev build

.PHONY: clean
clean:  ## Remove development artifacts
	rm -rf `find . -name .venv`
	rm -f `find . -name .coverage`
	rm -rf `find . -name .pdm-build`
	rm -rf `find . -name .pytest_cache`
	rm -rf `find . -name .ruff_cache`
	rm -rf `find . -name build`
	rm -rf `find . -name *.egg-info`
	rm -f `find . -name '*.pyc'`
	rm -f `find . -name '*.pyo'`
	rm -rf `find . ! -path '*/.git/*' -type d -empty`
	make -C cdk clean

# .PHONY: dev
# dev: dev-infra  ## Build and run each service locally
# 	uv run --package rex-api apps/rex-api/debug.py
#
# .PHONY: dev-docker
# dev-docker:  ## Build and run each service in a local docker container
# 	docker compose --profile=dev up --build --remove-orphans
#
# .PHONY: dev-infra
# dev-infra:  ## Start local infra dependency containers in background
# 	docker compose --profile=infra up --remove-orphans -d

.PHONY: install
install:  ## Install dependencies in .venv and refresh lockfile
	uv sync

.PHONY: install-locked
install-locked:  ## Install dependencies from lockfile in .venv
	uv sync --frozen

. PHONY: lock-check
lock-check:
	uv lock --check

.PHONY: format
format:  ## Format code overwriting if necessary
	uv run -- ruff format

.PHONY: format-check
format-check:
	uv run -- ruff format --check

.PHONY: lint
lint: format-check  ## Run static analysis checks for all libs and apps
	uv run -- ruff check
	uv run -- pyright

.PHONY: test
test: ## Run tests for all libs and apps
	uv run -- pytest -v --cov-report=term-missing --cov . -rxXs -m "not local"

.PHONY: test-integration
test-integration: ## Run integration tests
	uv run -- pytest -v --cov-report=term-missing --cov . -m "integration" -rxXs

.PHONY: test-all
test-all: ## Run unit and integration tests
	uv run -- pytest -v --cov-report=term-missing --cov . -m "not integration or integration" -rxXs

.PHONY: auth
auth: ## Setup default UV index to match pip
	mkdir -p ~/.config/uv
	echo "extra-index-url = ['${UV_INDEX}']" > ~/.config/uv/uv.toml






