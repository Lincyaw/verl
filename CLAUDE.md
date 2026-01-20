# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl is a flexible, efficient, and production-ready Reinforcement Learning (RL) training library for Large Language Models (LLMs) developed by ByteDance's Seed team. It's the open-source version of the HybridFlow framework presented at EuroSys 2025.

## Common Development Commands

### Testing
```bash
# Run CPU unit tests
pytest -s -x --asyncio-mode=auto tests/

# Run GPU unit tests (excludes CPU-only tests)
pytest -s -x --ignore-glob="*test_special_*.py" --ignore-glob='*on_cpu.py' --ignore-glob="*test_vllm*" --ignore-glob="*_sglang*" --ignore-glob="*_hf_rollout*" --ignore-glob="tests/models/" --ignore-glob='tests/special*' --ignore-glob="tests/experimental" --ignore-glob="tests/workers/reward_model" tests/

# Run a single test file
pytest -s -x tests/path/to/test_file.py

# Run sanity checks
pytest -s -x tests/special_sanity
```

### Code Quality


```bash
# Run all pre-commit hooks (linting, formatting, type checking)
pre-commit run --all-files

# Run ruff linter only
ruff check --fix --show-fixes --output-format=full .

# Run ruff formatter only
ruff format .

# Run mypy type checking
mypy .
```

### Building and Installation
```bash
# Install in development mode
pip install --no-deps -e .

# Install test dependencies
pip install -r requirements-test.txt

# Install main dependencies
pip install -r requirements.txt
```

### Documentation
```bash
# Build documentation
cd docs && make html

# View documentation
open docs/_build/html/index.html
```

## High-Level Architecture

### Core Design Pattern
verl follows a **single-controller multi-worker** pattern using Ray for distributed computing:

1. **TaskRunner** (verl/trainer/main_ppo.py) - Main orchestrator on driver node
2. **RayPPOTrainer** (verl/trainer/ppo/ray_trainer.py) - Core training logic coordinator
3. **Worker Groups** - Specialized workers for different tasks (actor, critic, reward, rollout)

### Key Components

**Data Layer (DataProto)**
- Central data structure built on PyTorch's TensorDict
- Separates tensor data from metadata for efficient Ray transfers
- Enables seamless data flow between distributed components

**Worker Architecture**
- **Actor Workers**: Generate rollouts and compute log probabilities
- **Critic Workers**: Compute value estimates
- **Reward Model Workers**: Compute rewards for training samples
- **Rollout Workers**: Handle text generation using vLLM/SGLang

**Backend Integration**
- **EngineRegistry** pattern supports multiple backends:
  - FSDPEngine (PyTorch FSDP)
  - MegatronEngine (NVIDIA Megatron-LM)
  - VeOmniEngine (custom backend)
  - MindspeedEngine (Huawei Ascend)

**Rollout Generation**
- Supports vLLM, SGLang, and HF Transformers backends
- Distributed across multiple GPUs for parallel generation

### Data Flow
1. Data loading through DataProto
2. Rollout generation by actor workers
3. Reward computation by reward models
4. Advantage estimation on driver
5. Policy update by actor/critic workers
6. Periodic checkpointing

### Important Patterns
- All workers inherit from base Worker class in verl/single_controller/
- Communication via Ray RPC with driver orchestration
- Backend-agnostic design through EngineRegistry
- Scales to 100B+ parameter models across hundreds of GPUs

## Key Files and Directories

- `verl/trainer/main_ppo.py` - Main entry point for PPO training
- `verl/trainer/ppo/ray_trainer.py` - Core PPO training orchestrator
- `verl/workers/` - Worker implementations
- `verl/workers/engine/` - Backend engine implementations
- `verl/single_controller/` - Ray-based distributed infrastructure
- `examples/` - Example training scripts for different algorithms
- `tests/` - Test suite organized by component

## Development Tips

- Always run pre-commit hooks before committing
- Tests are organized by component - check the appropriate test directory for your changes
- The project uses Ruff for linting/formatting (line length: 120)
- MyPy is used for type checking but many modules have errors ignored
- When adding new backends, follow the EngineRegistry pattern in verl/workers/engine/