.PHONY: clean check-clean test lint format release

# Code quality
lint:
	uv run ruff check gassolar/

# Code formatting
format:
	uv run ruff format gassolar/
	uv run ruff check --select I --fix gassolar/

# Testing
test:
	uv run pytest tests/ -v

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check-clean:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Found uncommitted changes:"; \
		git status --porcelain; \
		exit 1; \
	else \
		echo "Working directory is clean."; \
	fi

# Releasing (git tag + GitHub Release only -- gassolar isn't published to PyPI)
release: check-clean  # Cut a release: make release V=x.y.z
	@if [ -z "$(V)" ]; then \
		echo "Usage: make release V=x.y.z"; \
		exit 1; \
	fi
	@echo "$(V)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$$' || { \
		echo "V must look like x.y.z (got '$(V)')"; \
		exit 1; \
	}
	@git fetch origin main --quiet
	@if [ "$$(git rev-parse HEAD)" != "$$(git rev-parse origin/main)" ]; then \
		echo "HEAD is not up to date with origin/main. Pull or push first."; \
		exit 1; \
	fi
	gh release create v$(V) --generate-notes
