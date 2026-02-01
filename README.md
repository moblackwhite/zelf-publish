# ZELF

A data cleaning tool based on Bayesian Networks and Large Language Models.

## Overview

ZELF is an automated data cleaning tool that combines:
- **Bayesian Networks (BN)**: Models probabilistic dependencies between data attributes
- **Large Language Models (LLM)**: Enhances functional dependency detection and assists data repair

## Installation

### Requirements

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Environment Variables

Create a `.env` file:

```bash
# LLM API Configuration
OPENAI_API_KEY=your-api-key
LLM_API_BASE_URL=https://api.openai.com/v1  # Optional, for custom API
```

## Usage

```bash
python -m bn_llm.cli clean -d hospital --model gpt-4o --method HYBRID
```

## Experimental Results

Pre-computed experimental results are stored in `outputs/cache/` (committed to Git):

- **`outputs/cache/bn_structures/`**: Cached Bayesian network structures for each dataset
- **`outputs/cache/fd_results/`**: Functional dependency detection results from multiple LLM models (gpt-4o, claude-sonnet-4, gemini-3-flash, qwen3 variants, etc.)

These cached results enable reproducibility and faster experimentation.

## Project Structure

```
zelf/
├── src/bn_llm/           # Python package
│   ├── cli.py            # CLI entry point
│   ├── config.py         # Configuration management
│   ├── core/             # Core algorithms
│   ├── llm/              # LLM integration
│   ├── pipeline/         # Experiment pipelines
│   └── utils/            # Utility modules
├── configs/              # Configuration files
├── data/datasets/        # Datasets
└── outputs/              # Output directory
    └── cache/            # Cached results
```
