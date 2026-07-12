.PHONY: clean check-clean test lint format

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
