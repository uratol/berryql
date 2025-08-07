# Makefile for BerryQL development
.PHONY: help clean format lint test build install-dev check dist-test dist

# Default target
help:
	@echo "BerryQL Development Commands:"
	@echo "  make install-dev  - Install in development mode"
	@echo "  make format       - Format code with black and isort"
	@echo "  make lint         - Run linting tools"
	@echo "  make test         - Run tests"
	@echo "  make build        - Build the package"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make check        - Check package integrity"
	@echo "  make dist-test    - Build and upload to Test PyPI"
	@echo "  make dist         - Build and upload to PyPI"
	@echo "  make all          - Run format, lint, test, and build"

# Installation
install-dev:
	pip install -e .[dev,test]

# Code formatting
format:
	black berryql tests examples
	isort berryql tests examples

# Linting
lint:
	mypy berryql
	flake8 berryql tests examples

# Testing
test:
	pytest tests/ -v --cov=berryql --cov-report=html --cov-report=term

# Building
build: clean
	python -m build

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/

# Package checking
check: build
	python -m twine check dist/*

# Distribution
dist-test: build check
	python -m twine upload --repository testpypi dist/*

dist: build check
	python -m twine upload dist/*

# Complete workflow
all: format lint test build check

# Pre-commit setup
pre-commit:
	pre-commit install
