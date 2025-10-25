# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- custom field helper: add `read_only` argument (default: true) matching `field` semantics. When true, custom fields are excluded from mutation input types.

## [0.1.0] - 2025-08-14

- Reorganized requirements into core/dev with umbrella requirements.txt
- Updated pyproject metadata and optional extras for dev/test/adapters
- Added Python 3.13 classifier and modernized tooling versions
- Ensured MANIFEST includes essential docs and package sources
- Prepared packaging workflow (build, check, publish) via Makefile

