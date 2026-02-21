.PHONY: test lint

test:
	uv run pytest tests/ -v

lint:
	uv run flake8 gassolar/
