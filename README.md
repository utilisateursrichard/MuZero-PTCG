# PTCG MuZero — Reinforcement Learning Agent for Pokémon TCG

This project implements an agent based on the **MuZero** algorithm for the Kaggle **Pokémon TCG AI Battle** competition. The entire architecture is built in **JAX**, **Flax**, **Optax**, and **Mctx** to enable extremely fast execution and training on GPU acceleration (with multi-GPU support via `jax.pmap`).

---

## 🚀 Key Features

- **MuZero in JAX/Flax**: High-performance implementation of the MuZero planning algorithm (incorporating Representation, Dynamics, and Prediction networks).
- **Search Under Imperfect Information**: Uses an **ISMCTS** (Information Set Monte Carlo Tree Search) variant with belief-state determinization sampling to handle hidden information (e.g., opponent's hand, deck ordering).
- **Dual-GPU / Multi-GPU Training**: Paralleled training using `jax.pmap`. Network parameters and optimizer state are replicated across devices, and gradients are synchronized via `jax.lax.pmean`.
- **Co-Trained Deck Builder**: Jointly optimizes the starting 60-card deck composition via **REINFORCE** with an exponential moving average (EMA) baseline alongside the game-playing agent.
- **Interpretability & Diagnostics (Probing)**: Integrated diagnostic heads that predict hidden state features from MuZero's latent representations (e.g., presence of key cards in hand, remaining energy cards) to analyze representation quality.
- **HuggingFace Hub Integration**: Automatic checkpoint saving and configuration logging pushed directly to the HuggingFace Hub for seamless remote tracking during long training runs.
- **Anti-Freeze Security (Heartbeat Thread)**: A background heartbeat thread monitors training steps and automatically dumps active thread stack-traces if inactivity exceeds 60 seconds.

---

## 📁 Project Structure

```text
ptcg_muzero/
├── cards/
│   └── encoder.py          # Encoders for static card attributes from CSV data
├── env/
│   ├── __init__.py
│   ├── cabt_api.py         # Kaggle environment / cg-lib discovery and interfaing
│   ├── encoding.py         # Dense observation encoder for network consumption
│   └── wrapper.py          # Environment wrapper and self-play loops
├── interpretability/
│   └── probes.py           # Diagnostic heads and probing evaluation metrics
├── models/
│   ├── deck_builder.py     # DeckBuilderNetwork and deck sampling logic
│   └── networks.py         # MuZero network components (Flax models)
├── search/
│   └── ismcts.py           # ISMCTS implementation wrapped around Gumbel MuZero
├── training/
│   ├── activity.py         # Training activity tracking and thread dump utilities
│   ├── loss.py             # Value, Reward, Policy, Probes, and Deck REINFORCE loss functions
│   ├── replay_buffer.py    # Prioritized Replay Buffer
│   └── trainer.py          # Main training loop, JAX pmap config, and logging
├── config.py               # Central configuration file containing typed hyperparameters
├── main.py                 # Main entry point (CLI)
└── requirements.txt        # Python package dependencies
```

---

## 🛠️ Installation & Setup

### Prerequisites

This project requires Python 3.10+ and is optimized for GPU-accelerated execution with JAX.

### 1. Install Dependencies

Install the packages listed in `requirements.txt`:
```bash
pip install -r ptcg_muzero/requirements.txt
```

### 2. Install JAX with GPU Support (CUDA 12)

For GPU acceleration, install the appropriate JAX release:
```bash
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 3. Kaggle Data Setup

If running locally, ensure you have access to the card database (`EN_Card_Data.csv`) and the reference sample submission deck defined in your `InfraConfig`.

---

## ⚙️ Configuration (`config.py`)

All hyperparameters are grouped in typed dataclasses within `config.py`:

- **`ModelConfig`**: MuZero network architecture (latent dimension size, Transformer layers, attention heads) and game boundary constants (max hand size, bench size, discard pile size).
- **`SearchConfig`**: ISMCTS details (simulation counts, belief determinization count `num_belief_samples`, Gumbel action constraints, Dirichlet noise).
- **`TrainConfig`**: Optimizer params, loss weights, replay buffer specifications, and training step intervals.
- **`HFConfig`**: Configuration settings for syncing checkpoints directly to the HuggingFace Hub.
- **`InfraConfig`**: Device assignments, random seed, local directory paths for logging and saving.

---

## 🕹️ CLI Usage (`main.py`)

The `ptcg_muzero/main.py` entrypoint exposes the following commands:

### 1. Training (`train`)
Launches the joint MuZero and Deck Builder training process.
```bash
python ptcg_muzero/main.py train --devices 2 --hf-repo "your-username/ptcg-muzero"
```
*Key Options:*
- `--config <path>`: Load a custom JSON configuration file.
- `--no-hf`: Disable automated checkpoint pushes to the HuggingFace Hub.
- `--debug`: Disable JAX JIT compilation for debugging purposes.

### 2. Evaluation (`eval`)
Evaluates a specific checkpoint by matching the agent against an opponent making random decisions.
```bash
python ptcg_muzero/main.py eval --ckpt ./checkpoints/checkpoint_step_10000.safetensors --eval-games 20
```

### 3. Probing Diagnostic (`probe-diag`)
Runs diagnostic evaluations and outputs accuracy metrics for the latent state probing classifiers on a checkpoint.
```bash
python ptcg_muzero/main.py probe-diag --ckpt ./checkpoints/checkpoint_step_10000.safetensors
```

### 4. Kaggle Submission (`submit`)
Generates a standalone, submission-ready file `agent_submit.py` for Kaggle. It is pre-configured to download the model weights from your HuggingFace repository.
```bash
python ptcg_muzero/main.py submit --hf-repo "your-username/ptcg-muzero"
```

---

## 🧠 Technical Highlights

### Card Representations (`encoder.py`)
Pokémon cards contain detailed static features (HP, evolutionary stage, attack costs, damage outputs). These are represented in a static matrix mapping (`CardStaticFeatures`). At runtime, card features are combined with learnable embeddings to construct inputs for the MuZero representation network.

### ISMCTS (`ismcts.py`)
Since card hands and decks are hidden from the active player, the search process determinizes the hidden environment state. It draws random card distributions corresponding to the remaining deck composition before evaluating candidate actions using a JAX-friendly Gumbel MuZero simulation.

### Optimization & Loss Structure (`loss.py`)
The joint loss optimizes both gameplay and deck composition:
$$\mathcal{L} = \mathcal{L}_{\text{policy}} + c_1 \mathcal{L}_{\text{value}} + c_2 \mathcal{L}_{\text{reward}} + c_3 \mathcal{L}_{\text{probes}} + c_4 \mathcal{L}_{\text{deck}}$$

The deck building sub-network is optimized using REINFORCE, scaling gradients based on game outcomes (win/loss) offset by a running average baseline.
