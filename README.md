# ⚡ Pokémon TCG MuZero (PTCG-MuZero)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![JAX / Flax](https://img.shields.io/badge/JAX%20%2F%20Flax-0.4.25%2B-red?style=for-the-badge&logo=google)](https://github.com/google/jax)
[![mctx](https://img.shields.io/badge/mctx-DeepMind%20MCTS-blue?style=for-the-badge&logo=deepmind)](https://github.com/google-deepmind/mctx)
[![Modal Cloud](https://img.shields.io/badge/Modal-GPU%20Training-00C7B7?style=for-the-badge&logo=modal)](https://modal.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Checkpoints-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracking-FFBE00?style=for-the-badge&logo=weightsandbiases)](https://wandb.ai/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-PTCG%20AI%20Battle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)

---

## 📖 Overview

**PTCG-MuZero** is a state-of-the-art Deep Reinforcement Learning agent engineered for the **Kaggle Pokémon TCG AI Battle Competition**.

The Pokémon Trading Card Game (PTCG) presents substantial challenges for competitive AI:
1. **Imperfect Information**: Hidden opponent hand, facedown Prize cards, and unknown deck order.
2. **Stochastic Transitions**: Coin flips, deck shuffling, and random card draws.
3. **Combinatorial & Dynamic Action Spaces**: Contextual decision trees (attacks, retreat decisions, energy attachments, Trainer item/supporter activations, bench/deck search targeting).
4. **Long Decision Horizons**: Multi-turn resource tempo, prize trade management, and card advantage mechanics.

This system adapts **MuZero** (DeepMind) by combining it with **Information Set Monte Carlo Tree Search (ISMCTS)**, a **Transformer Encoder backbone** with static & learned card embeddings, and enhancements inspired by **EfficientZero V2** and **SpeedyZero**.

---

## 🏛️ System Architecture

```
                                 ┌────────────────────────────────────────┐
                                 │         Game Observation (cabt)        │
                                 └──────────────────┬─────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. h(s) — Representation Network (Transformer Encoder)                                                 │
│    • Static Features (48D) + Learned Embeddings (64D) = 112D per card                                 │
│    • Encodes board state (Active, Bench, Hand, Discard, Prizes, Energies, Status, Legal Options)       │
│    • Multi-Head Self-Attention (4 layers, 8 heads, dim=256) → [CLS] token = Latent State z             │
└───────────────────────────────────┬────────────────────────────────────────────────────────────────────┘
                                    │
                                    ├───► Mechanistic Interpretability Probes (KO, Type, Lead, Hand, Energy)
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. ISMCTS + Gumbel MuZero Search (mctx)                                                                │
│    • Belief Sampling: Determinizes opponent hand/deck from known legal deck distribution               │
│    • Gumbel MuZero Policy: Provably sound policy improvement under constrained simulation budgets      │
│    • Action Selection: Dynamic temperature annealing for adaptive exploration/exploitation             │
└───────────────────┬─────────────────────────────────────────────────┬──────────────────────────────────┘
                    │                                                 │
                    ▼ (f: Prediction)                                 ▼ (g: Dynamics)
┌───────────────────────────────────────┐         ┌──────────────────────────────────────────────────────┐
│ f(z) → (π, v)                         │         │ g(z, a) → (z_next, r)                                │
│ • Policy logits over 128 action slots │         │ • Latent state transition + intermediate reward      │
│ • Categorical Two-Hot Value (51 bins) │         │ • Collapsed-expectation for stochasticity            │
└───────────────────────────────────────┘         └──────────────────────────────────────────────────────┘
```

### The MuZero Triad Networks

1. **Representation Network $h(s) \to z$**:
   - Compresses the complete game state into a continuous latent vector $z \in \mathbb{R}^{256}$.
   - Every card token combines a 48-dimensional static feature vector extracted from `EN_Card_Data.csv` (Card Type, HP, Attack costs, Damage, Weakness/Resistance, Evolution Stage, Rules) with a 64-dimensional learned embedding.
   - Board entities are processed through a Transformer Encoder with attention masking over vacant slots.

2. **Prediction Network $f(z) \to (\pi, v)$**:
   - **Policy $\pi(a|z)$**: Probability distribution over $128$ action options (illegal actions strictly masked to $-\infty$).
   - **Value $v(z)$**: Categorical distribution over 51 discrete bins spanning $[-1.8, +1.8]$ using two-hot continuous representation to reduce gradient variance.

3. **Dynamics Network $g(z, a) \to (z', r)$**:
   - Predicts next latent state $z_{k+1}$ and intermediate reward $r_{k+1}$ conditioned on action $a$.
   - Handles stochastic transitions (e.g. coin flips) via collapsed-expectation approximation.

---

## 🚀 Key Features

- **⚡ JAX / XLA Multi-GPU Acceleration**: End-to-end vectorization, just-in-time compilation, and distributed data parallelism via `jax.pmap`.
- **🎲 Hybrid Information Set MCTS (ISMCTS)**: Belief sampling determinizes hidden opponent cards directly from the active, legal deck distribution rather than hallucinated global cards.
- **📈 EfficientZero V2 & SpeedyZero Advancements**:
  - **Consistency Loss**: Enforces consistency between unrolled dynamics $z_{k+1} = g(z_k, a_k)$ and actual observation representation $h(s_{k+1})$.
  - **Prioritized Experience Replay (PER)**: Proportional TD-error sampling with annealed Importance Sampling corrections ($\alpha, \beta$).
  - **Polyak Target Networks**: Exponential Moving Average ($\tau = 0.995$) for bootstrap target stability.
  - **In-Pipeline GPU Reanalyze & Priority Refresh**: Continuous MCTS target recomputation directly on GPU.
  - **Surgical Head Reset & Representation Warming**: Ability to reset value/policy heads while preserving the 94%+ representation space $h(s)$ with progressive unfreezing ramps.
- **🎯 Competitive Reward Shaping**:
  - Progressive rewards per Prize taken ($+1/12 \approx +0.0833$) and conceded ($-1/12$).
  - Terminal game outcomes: Win ($+1.0$) / Loss ($-1.0$).
  - Additional penalty for Deck-out losses ($-0.10$).
- **🎴 Neural Deck Builder**:
  - Policy gradient optimization (REINFORCE) for generating 60-card tournament-legal decks under game constraints (Basic Pokémon requirements, energy quotas, 4-copy limits, single Ace Spec restriction).
  - One-click export to official `deck.csv` and interactive vector graphic `deck.svg`.
- **🔬 Mechanistic Interpretability & Latent Probing**:
  - 5 independent linear probes trained on detached latent vectors (`jax.lax.stop_gradient`) to monitor strategic concept encoding:
    - `active_in_ko_range`: Risk of active Pokémon being KO'd in 1 hit.
    - `type_advantage`: Weakness and resistance advantage.
    - `prize_lead`: Prize card advantage status.
    - `hand_advantage`: Hand card advantage.
    - `opp_energy_ready`: Opponent active attack readiness.
  - Standalone 2D dimensionality reduction tool (`visualize_latent.py`) via UMAP, t-SNE, and PCA.
- **☁️ Cloud Training on Modal & HuggingFace Hub Sync**:
  - 1-command serverless training on NVIDIA L4 GPUs via `modal_train.py`.
  - Automatic asynchronous checkpoint synchronization (`safetensors`, `config.json`) and replay buffer backups on Hugging Face Hub.
  - Live telemetry and metrics logging via Weights & Biases (WandB).
- **📦 Turnkey Kaggle Submission Pipeline**:
  - Autonomous submission generator packaging `main.py`, `deck.csv`, `muzero.safetensors`, and vendored runtime dependencies (`mctx`, `chex`, `cg`).

---

## 📁 Repository Structure

```
.
├── agent_submit.py             # Autonomous inference agent for Kaggle
├── checkpoints/                # Local checkpoint directory (.pkl, config.json)
├── competiton/                 # Kaggle competition assets (cards, engine)
│   ├── EN_Card_Data.csv        # Official card database
│   └── sample_submission/      # Reference cg-lib engine
├── deck.csv                    # Active competition deck (60 card IDs)
├── deck.svg                    # Visual representation of the deck
├── export_deck.py              # Deck sampler and exporter (CSV & SVG)
├── export_deck_svg.py          # SVG card layout renderer
├── modal_train.py              # Cloud training script on Modal (GPU L4)
├── ptcg_muzero/                # Core MuZero package
│   ├── cards/
│   │   └── encoder.py          # 48D static features and card embeddings
│   ├── config.py               # Central configuration dataclasses
│   ├── env/
│   │   ├── baselines.py        # Benchmark agents (Greedy, Random, Heuristics)
│   │   ├── cabt_api.py         # C++/Python engine bridge discovery
│   │   ├── encoding.py         # Board state vectorization
│   │   └── wrapper.py          # Environment wrapper & parallel self-play
│   ├── export/
│   │   └── hub.py              # HuggingFace Hub synchronization
│   ├── interpretability/
│   │   └── probes.py           # Linear probing classifiers
│   ├── models/
│   │   ├── deck_builder.py     # REINFORCE deck optimization network
│   │   └── networks.py         # Flax Linen implementation of h, f, g
│   ├── search/
│   │   └── ismcts.py           # Imperfect information MCTS & Gumbel search
│   ├── training/
│   │   ├── activity.py         # Heartbeat monitoring & deadlock detector
│   │   ├── loss.py             # Vectorized MuZero loss functions
│   │   ├── replay_buffer.py    # Prioritized replay buffer (PER) & TD returns
│   │   └── trainer.py          # Multi-GPU training loop & JIT orchestration
│   ├── analyze_attack_transitions.py # Energy attachment & attack delay analytics
│   ├── diag_prize.py           # Engine prize state validator
│   ├── evaluation.py           # Evaluation metrics & 95% Confidence Intervals
│   ├── local_probe_diag.py     # Local probe accuracy checker
│   ├── main.py                 # Central CLI entrypoint
│   ├── requirements.txt        # Python dependencies
│   └── submit_main.py          # Standalone Kaggle submission template
├── submission/                 # Built submission folder
├── submission.tar.gz           # Packaged Kaggle submission archive
└── visualize_latent.py         # 2D Latent space visualization (UMAP / t-SNE / PCA)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10 or 3.11
- **Hardware**: NVIDIA GPU with CUDA 12 support

### 1. Clone the Repository & Create Virtual Environment

```bash
git clone https://github.com/your-username/ptcg-muzero.git
cd ptcg-muzero

# Create virtual environment
python3 -m venv .venv_gpu
source .venv_gpu/bin/activate
```

### 2. Install Dependencies

```bash
# Install JAX with CUDA 12 support
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install project requirements
pip install -r ptcg_muzero/requirements.txt
```

### 3. Environment Variables (Recommended)

```bash
export HF_TOKEN="hf_your_secret_token"        # For automated HuggingFace backups
export WANDB_API_KEY="your_wandb_key"         # For live telemetry and metric tracking
```

---

## 💻 CLI Usage Guide

All operations are managed through the central entrypoint: `ptcg_muzero/main.py`.

### 1. 🏋️ Training (`train`)

#### Local Multi-GPU Training
```bash
python ptcg_muzero/main.py train --devices 2
```

#### Cloud Training on Modal (NVIDIA L4 GPU)
```bash
# First-time Modal setup
modal setup

# Launch in detached mode (persisting checkpoints automatically)
modal run --detach modal_train.py
```

#### Resuming from HuggingFace Checkpoints
```bash
# Resume from latest HF checkpoint
python ptcg_muzero/main.py train -s HF

# Resume from a specific training step
python ptcg_muzero/main.py train -s hf@135000

# Surgical Value Head Reset with clean Replay Buffer
python ptcg_muzero/main.py train -s hf@135000 --reset-value-head --fresh-buffer
```

---

### 2. ⚔️ Strength Benchmarking (`test`)

The `test` mode provides statistical validation against external baselines with **95% Confidence Intervals** (neutralizing first-player advantage via side alternation).

#### Benchmark Against the Greedy Heuristic Baseline
```bash
# 40 games against full-strength Greedy agent
python ptcg_muzero/main.py test -s HF -o greedy -n 40 --sims 25
```

#### Tuning Difficulty with $\epsilon$-Greedy
```bash
# epsilon=0.0 (full strength) -> epsilon=1.0 (pure random baseline)
python ptcg_muzero/main.py test -s HF -o greedy -e 0.3 -n 40
```

#### Historical Progression Benchmark (Checkpoint vs Checkpoint)
```bash
# Current checkpoint vs earlier step (e.g. Step 150k vs Step 100k)
python ptcg_muzero/main.py test -s hf@150000 -o checkpoint --opponent-weights hf@100000 -n 40
```

---

### 3. 🔬 Interpretability & Diagnostics

#### Linear Probes Verification
```bash
python ptcg_muzero/main.py probe-diag --ckpt checkpoints/ckpt_latest.pkl
```

#### 2D Latent Space Projection (UMAP / t-SNE / PCA)
Generates high-resolution visualization (`latent_space.png`) correlating latent dynamics $z$ with game states and predictions:
```bash
python visualize_latent.py --safetensors checkpoints/muzero.safetensors --method umap --games 10
```

#### Energy & Attack Sequence Diagnostics
```bash
python ptcg_muzero/analyze_attack_transitions.py
```

---

### 4. 🎴 Deck Builder (Sampling & SVG Export)

Sample a 60-card legal deck from trained `DeckBuilderNetwork` parameters:

```bash
python export_deck.py --safetensors checkpoints/deck_builder.safetensors --csv deck.csv --svg deck.svg
```

---

### 5. 🚀 Kaggle Submission Preparation (`submit`)

Generates a standalone submission package with vendored libraries, weights, and configurations:

```bash
python ptcg_muzero/main.py submit -s HF --out-dir submission

# Create archive ready for Kaggle submission
tar -czvf submission.tar.gz -C submission .
```

---

## ⚙️ Hyperparameters & Configuration

Configurations are typed dataclasses defined in `ptcg_muzero/config.py`:

| Category | Hyperparameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **Model** | `latent_dim` | `256` | Latent space vector dimension $z$ |
| | `num_enc_layers` | `4` | Transformer Encoder layer count |
| | `num_heads` | `8` | Multi-head attention count |
| | `card_embed_dim` | `64` | Learned card embedding dimension |
| | `max_actions` | `128` | Maximum action/option slots |
| | `num_value_bins` | `51` | Value categorical bins (Two-Hot support) |
| **Search** | `num_simulations` | `25` | MCTS simulation budget per decision |
| | `num_belief_samples`| `4` | ISMCTS belief determinizations |
| | `max_num_considered_actions` | `8` | Considered actions in Sequential Halving |
| | `temperature_init` | `0.80` | Initial exploration temperature |
| | `temperature_min` | `0.15` | Minimum endgame temperature floor |
| **Train** | `num_unroll_steps` | `5` | Recurrent dynamics unroll steps $K$ |
| | `td_steps` | `20` | Multi-step bootstrap target horizon |
| | `batch_size` | `64` | Total batch size across all GPUs |
| | `learning_rate` | `3e-4` | Peak learning rate (AdamW with warmup) |
| | `replay_buffer_size`| `200_000` | Prioritized Replay Buffer capacity |
| | `min_replay_size` | `10_000` | Replay warmup size before training |
| | `target_network_tau`| `0.995` | Polyak Target Network momentum |
| | `consistency_loss_weight` | `2.0` | Weight for $h(s) \approx g(z, a)$ loss |
| | `policy_entropy_weight` | `0.02` | Policy entropy exploration bonus |

---

## 📊 Telemetry & Training Metrics

Training progress is monitored via an asynchronous **heartbeat thread** and logged to **Weights & Biases**:
- **Normalized Policy Entropy ($H_{norm}$)** & Max Choice Probability ($p_{max}$).
- **Action Type Distribution**: % Attacks, % Energy Attachments, % End Turn decisions.
- **Probe Classification Accuracies**: Semantic understanding of game concepts in latent space $z$.
- **Win Rates & Margin Stats**: Statistical evaluations with 95% Confidence Intervals.
- **Representation Gradient Scale**: Tracking progress during representation unfreezing ramps.

---

## 📜 License & Acknowledgments

- Developed for the **Kaggle Pokémon TCG AI Battle Competition**.
- Based on foundational research:
  - *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (Schrittwieser et al., Nature 2020)
  - *Policy improvement by planning with Gumbel* (Danihelka et al., ICLR 2022)
  - *Mastering Diverse Domains through World Models - EfficientZero V2* (Ye et al., 2024)
- Pokémon TCG game engine licensed under Kaggle Competition rules.
