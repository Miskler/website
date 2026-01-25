.PHONY: help lint format type-check clean all test

help:
	@echo "Available commands:"
	@echo "  make lint            - Run linting"
	@echo "  make format          - Format code"
	@echo "  make type-check      - Run type checking"
	@echo "  make clean           - Clean build artifacts"
	@echo "  make all             - Run format + lint + type-check"

# Вариант 1: Используем конфигурацию в pyproject.toml
lint:
	flake8 . --exclude=.venv
	black --check .
	isort --check-only .

format:
	black .
	isort .

type-check:
	mypy .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: format lint type-check