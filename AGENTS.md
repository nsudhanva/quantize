# AGENTS.md

This file provides guidance for AI agents working on this repository.

## Project Overview

This project demonstrates various strategies to compress a PyTorch model for efficient inference. It also showcases optimizations like operator fusion and key-value caching. The goal is to provide clear examples of different model compression techniques.

## Technology Stack

*   **Language:** Python 3.11+
*   **Framework:** PyTorch
*   **Dependency Management:** uv
*   **Testing:** pytest
*   **Linting/Formatting:** ruff, pre-commit

## Development Environment

To set up the development environment, you'll need to have `uv` installed.

1.  **Install dependencies:**
    ```bash
    uv sync --group dev
    ```
    This command will install all the necessary dependencies, including those for development, as specified in `pyproject.toml`.

## Key Scripts

There are three main scripts in this project:

*   `train.py`: Trains a model and exports several compressed variants.
    ```bash
    uv run python train.py
    ```
*   `fusion.py`: Loads a trained model (`model.pth`) and applies operator fusion using TorchScript.
    ```bash
    uv run python fusion.py
    ```
*   `kv_cache.py`: Demonstrates the use of a key-value cache in an attention mechanism.
    ```bash
    uv run python kv_cache.py
    ```

## Testing

The project uses `pytest` for unit testing.

*   **Run all tests:**
    ```bash
    uv run pytest
    ```
    Ensure that all tests pass before submitting any changes.

## Linting and Formatting

This project uses `ruff` for linting and formatting, managed through `pre-commit`.

*   **Run on staged files:**
    ```bash
    uv run pre-commit run --all-files
    ```
*   The pre-commit hooks are also configured to run automatically in GitHub Actions, but it is good practice to run them locally before committing.

## General Guidelines

*   When adding new features, please also add corresponding tests.
*   Ensure that any new dependencies are added to the `pyproject.toml` file.
*   Keep the `README.md` file up-to-date with any changes to the project's functionality or structure.
