#!/usr/bin/env python3
"""
Development tasks for BerryQL.

Use this helper instead of the old build.py to avoid
shadowing the PyPA `build` module.
"""

import subprocess
import sys


def run_command(command, check=True):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0


def clean():
    print("Cleaning build artifacts...")
    # Cross-platform cleaning via Python
    import shutil, os

    for path in ["build", "dist", ".pytest_cache", "htmlcov"]:
        shutil.rmtree(path, ignore_errors=True)
    # Remove egg-info dirs
    for name in os.listdir("."):
        if name.endswith(".egg-info"):
            shutil.rmtree(name, ignore_errors=True)
    # Bytecode caches
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass
    print("Clean completed.")


def format_code():
    print("Formatting code...")
    run_command("black berryql tests examples")
    run_command("isort berryql tests examples")
    print("Code formatting completed.")


def lint():
    print("Running linting...")
    success = True
    if not run_command("mypy berryql", check=False):
        success = False
    if not run_command("flake8 berryql tests examples", check=False):
        success = False
    if not success:
        print("Linting failed.")
        sys.exit(1)
    print("Linting passed.")


def test():
    print("Running tests...")
    run_command("pytest tests/ -v --cov=berryql --cov-report=html --cov-report=term")
    print("Tests completed.")


def build():
    print("Building package...")
    clean()
    run_command("python -m build")
    print("Build completed.")


def install_dev():
    print("Installing in development mode...")
    run_command("pip install -e .[dev,test]")
    print("Development installation completed.")


def publish_test():
    print("Publishing to Test PyPI...")
    run_command("python -m twine upload --repository testpypi dist/*")
    print("Test publication completed.")


def publish():
    print("Publishing to PyPI...")
    run_command("python -m twine upload dist/*")
    print("Publication completed.")


def check_package():
    print("Checking package...")
    run_command("python -m twine check dist/*")
    run_command("check-manifest")
    print("Package check completed.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python dev_tasks.py <command>")
        print("Commands: clean, format, lint, test, build, install-dev, check, publish-test, publish, all")
        sys.exit(1)
    command = sys.argv[1]
    commands = {
        "clean": clean,
        "format": format_code,
        "lint": lint,
        "test": test,
        "build": build,
        "install-dev": install_dev,
        "check": check_package,
        "publish-test": publish_test,
        "publish": publish,
        "all": lambda: (format_code(), lint(), test(), build(), check_package()),
    }
    fn = commands.get(command)
    if not fn:
        print(f"Unknown command: {command}")
        sys.exit(1)
    fn()


if __name__ == "__main__":
    main()
