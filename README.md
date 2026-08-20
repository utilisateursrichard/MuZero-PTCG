# Pokémon TCG MuZero (PTCG-MuZero)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![JAX / Flax](https://img.shields.io/badge/JAX%20%2F%20Flax-0.4.25%2B-red?style=for-the-badge&logo=google)](https://github.com/google/jax)
[![mctx](https://img.shields.io/badge/mctx-DeepMind%20MCTS-blue?style=for-the-badge&logo=deepmind)](https://github.com/google-deepmind/mctx)
[![Modal Cloud](https://img.shields.io/badge/Modal-GPU%20Training-00C7B7?style=for-the-badge&logo=modal)](https://modal.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Checkpoints-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracking-FFBE00?style=for-the-badge&logo=weightsandbiases)](https://wandb.ai/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-PTCG%20AI%20Battle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)

---

## What is PTCG MuZero?

PTCG MuZero (or pokémon trading card game MuZero) is a deep Neural network (NN) based on the MuZero architecture made by google deepmind and optimizations from Stochastic MuZero, EfficientZero, EfficientZero V2, and Gumbel MuZero.

---

## System Architecture

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

##  How to use :

### Prerequisites
- **Operating System**: Linux, Windows or MacOS (created originaly for linux, some feature might be a little buggy on windows or macOS)
- **Python**: 3.10, 3.11 or 3.12 (the recent version of python (e.g. 3.14 et 3.13) will probably not work)
- **Hardware**: Any GPU supporting Vulkan 1.3+ with Compute capabilites or a CPU (we reccommand using a GPU for speed)

## To train a PTCG-MuZero :
### 1. git clone the project 

```bash
git clone https://github.com/utilisateursrichard/ptcg-muzero.git
cd ptcg-muzero
```

### 2. Install Dependencies

```bash
# Install project requirements
pip install -r ptcg_muzero/requirements.txt
```

### 3. Environment Variables (optionnal)

```bash
export HF_TOKEN="hf_your_secret_token"        # if you use HF (recommended)
export WANDB_API_KEY="your_wandb_key"         # if you use W&B
```
### 4. start the training 

```bash
python ptcg_muzero/main.py train --devices <Number of GPU/TPU thath you have>
```
or if you use modal : 
```bash
# First-time Modal setup
modal setup

# Launch in detached mode (persisting checkpoints automatically)
modal run modal_train.py
```
## How to fight against the pretrained MuZero (available at richard151111/muzero-V2)

### 1. git clone the project 

```bash
git clone https://github.com/utilisateursrichard/ptcg-muzero.git
cd ptcg-muzero
```

### 2. Install Dependencies

```bash
# Install project requirements
pip install -r ptcg_muzero/requirements.txt
```
### 3. download form HF and compile it with IREE

```bash
# For Vulkan GPU (AMD / NVIDIA / Intel):
python export_iree.py -m HF -o muzero_vulkan.vmfb --target vulkan
# Or for CPU:
python export_iree.py -m HF -o muzero_CPU.vmfb --target cpu
```
note : be sure that you use the default config file

### 4. start the local server 

```bash
python battle_server.py --port 8000
```

### 5. to use it go to http://localhost:8000 and you should find the server


---

## How to test the model 

to test the model, you can use `python main.py test` to test it with different  ways :

### 1. Against a greedy heuristic baseline

to use it, use the following command :
```bash
python ptcg_muzero/main.py test -s HF -o greedy -n 40 --sims 25
```
in that command, -s HF is the model you wnat to test (in that case the latest model in the HF repository), -o the opponent (here greedy) -n is the number of games and the --sims is the number of MCTS node that you want (the more you add, the more the model will think)

### 2. Against a random agent

it is the same command as before but with the random arg (in -o) instead of greedy
```bash
python ptcg_muzero/main.py test -s HF -o random -n 40 --sims 25
```

### 3. against himself/ another model

for agaisnt himself you can use the self argument.
```bash
python ptcg_muzero/main.py test -s HF -o self -n 40 --sims 25
```
or you can use the checkpoint of the model you want to test
```bash
python ptcg_muzero/main.py test -s HF -o checkpoint --opponent-weights HF@150000 -n 40 --sims 25
```
in that case, it is the latest model (-s HF) aigainst another checkpoint (-o checkpoint) and the weight of the other checkpoints are the step 150000 (--opponent-weights HF@150000)
---

## Interpretability

to verify if the latent space of the model is healthy, you can use linear probes or visualize the latent space
1. just look at the W&B/ training logs to see the probes accuracy and loss
2. create a 2D visualization of a the latent space with the following commands :
```bash
python ptcg_muzero/visualize_latent.py --safetensors HF --method {tsne,pca,umap,phate} --games 10
```
you can choose between tsne, pca, umap or phate (my personnal favorite are phate and umap but they all are good and bad in some ways)

---


## Hyperparameters and configuration

Configurations are defined in `ptcg_muzero/config.py`:

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

## note from the dev :
my original language is French, some feature might be in french and some of my commit messages as well, but I tried to translate most of the feature taht you will probably use.

## License & Acknowledgments

- Developed for the **Kaggle Pokémon TCG AI Battle Competition**.
- Based on foundational research:
  - *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (Schrittwieser et al., Nature 2020)
  - *Policy improvement by planning with Gumbel* (Danihelka et al., ICLR 2022)
  - *Mastering Diverse Domains through World Models - EfficientZero V2* (Ye et al., 2024)
- Pokémon TCG game engine licensed under Kaggle Competition rules.
