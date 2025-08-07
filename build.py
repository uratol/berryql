#!/usr/bin/env python3
"""
Build script for BerryQL package.

This script handles common development tasks like building, testing, and publishing.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0


def clean():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    artifacts = [
        "build/",
        "dist/",
        "*.egg-info",
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        ".pytest_cache",
        ".coverage",
        "htmlcov/",
    ]
    
    for pattern in artifacts:
        run_command(f"rm -rf {pattern}", check=False)
    
    print("Clean completed.")


def format_code():
    """Format code using black and isort."""
    print("Formatting code...")
    run_command("black berryql tests examples")
    run_command("isort berryql tests examples")
    print("Code formatting completed.")


def lint():
    """Run linting tools."""
    print("Running linting...")
    success = True
    
    # Type checking with mypy
    if not run_command("mypy berryql", check=False):
        success = False
    
    # Linting with flake8
    if not run_command("flake8 berryql tests examples", check=False):
        success = False
    
    if success:
        print("Linting passed.")
    else:
        print("Linting failed.")
        sys.exit(1)


def test():
    """Run tests."""
    print("Running tests...")
    run_command("pytest tests/ -v --cov=berryql --cov-report=html --cov-report=term")
    print("Tests completed.")


def build():
    """Build the package."""
    print("Building package...")
    clean()
    run_command("python -m build")
    print("Build completed.")


def install_dev():
    """Install package in development mode."""
    print("Installing in development mode...")
    run_command("pip install -e .[dev,test]")
    print("Development installation completed.")


def publish_test():
    """Publish to Test PyPI."""
    print("Publishing to Test PyPI...")
    run_command("python -m twine upload --repository testpypi dist/*")
    print("Test publication completed.")


def publish():
    """Publish to PyPI."""
    print("Publishing to PyPI...")
    run_command("python -m twine upload dist/*")
    print("Publication completed.")


def check_package():
    """Check package integrity."""
    print("Checking package...")
    run_command("python -m twine check dist/*")
    run_command("check-manifest")
    print("Package check completed.")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python build.py <command>")
        print("Commands:")
        print("  clean        - Clean build artifacts")
        print("  format       - Format code with black and isort")
        print("  lint         - Run linting tools")
        print("  test         - Run tests")
        print("  build        - Build the package")
        print("  install-dev  - Install in development mode")
        print("  check        - Check package integrity")
        print("  publish-test - Publish to Test PyPI")
        print("  publish      - Publish to PyPI")
        print("  all          - Run format, lint, test, and build")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "clean":
        clean()
    elif command == "format":
        format_code()
    elif command == "lint":
        lint()
    elif command == "test":
        test()
    elif command == "build":
        build()
    elif command == "install-dev":
        install_dev()
    elif command == "check":
        check_package()
    elif command == "publish-test":
        publish_test()
    elif command == "publish":
        publish()
    elif command == "all":
        format_code()
        lint()
        test()
        build()
        check_package()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
