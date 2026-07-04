"""
ptcg_muzero/config.py
=====================
Central configuration for the PTCG MuZero agent.
All hyperparameters live in typed dataclasses so they can be passed cleanly,
serialised to JSON, and embedded in HuggingFace model cards.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Fixed game constants ──────────────────────────────────────────────────────
NUM_ENERGY_TYPES: int = 11   # C G R W L P F D M N Y (no "none" here)
NUM_STAGES: int = 7          # Basic, S1, S2, BasicEnergy, SpecEnergy, Trainer, Unknown
# CARD_STATIC_DIM is defined in cards/encoder.py and equals 48


@dataclass
class ModelConfig:
    # ── Card representation ────────────────────────────────────────────────
    num_card_ids: int = 600      # safe upper-bound; set to max(CSV card_id) + 1 at runtime
    card_embed_dim: int = 64     # learnable part
    # static part = 48 (CARD_STATIC_DIM); total = 112

    # ── Transformer backbone ───────────────────────────────────────────────
    latent_dim: int = 256
    num_heads: int = 8
    ff_dim: int = 512
    num_enc_layers: int = 4
    dropout_rate: float = 0.0    # kept 0 during self-play; set at training

    # ── Action / option space ──────────────────────────────────────────────
    max_actions: int = 128       # padded; invalid slots masked to -∞

    # ── Board padding constants ────────────────────────────────────────────
    max_hand_size: int = 12
    max_bench_size: int = 5
    max_discard_size: int = 60
    max_prize_size: int = 6

    # ── Interpretability ──────────────────────────────────────────────────
    num_probe_tasks: int = 5

    # ── Deck builder ──────────────────────────────────────────────────────
    num_deck_slots: int = 60


@dataclass
class SearchConfig:
    num_simulations: int = 50
    # ISMCTS: number of determinizations sampled at each root
    num_belief_samples: int = 4
    # Gumbel MuZero: number of considered actions at each node
    max_num_considered_actions: int = 16
    # Dirichlet noise at root (self-play exploration)
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


@dataclass
class TrainConfig:
    # ── MuZero unrolling ──────────────────────────────────────────────────
    num_unroll_steps: int = 5
    td_steps: int = 10
    gamma: float = 0.997

    # ── Optimiser ─────────────────────────────────────────────────────────
    batch_size: int = 256        # total across all devices
    learning_rate: float = 3e-4
    lr_warmup_steps: int = 2_000
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # ── Replay buffer ─────────────────────────────────────────────────────
    replay_buffer_size: int = 200_000
    min_replay_size: int = 10_000
    replay_alpha: float = 0.5    # priority exponent (0 = uniform, 1 = full priority)
    replay_beta: float = 0.4     # IS correction exponent (annealed toward 1 during training)

    # ── Training schedule ─────────────────────────────────────────────────
    num_total_steps: int = 500_000
    self_play_interval: int = 100    # global steps between self-play batches
    games_per_self_play: int = 8

    checkpoint_every: int = 1_000
    eval_every: int = 5_000
    eval_games: int = 20

    # ── Loss weights ──────────────────────────────────────────────────────
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    probe_loss_weight: float = 0.05
    deck_loss_weight: float = 0.1
    consistency_loss_weight: float = 2.0

    # ── Deck builder ──────────────────────────────────────────────────────
    deck_lr: float = 1e-3
    deck_entropy_coef: float = 0.01
    deck_baseline_ema: float = 0.99    # exponential moving average for REINFORCE baseline


@dataclass
class HFConfig:
    enabled: bool = True
    repo_id: str = "richard151111/muZero"
    private: bool = True
    # Nom de la variable d'environnement contenant le token HF
    # (ex: export HF_TOKEN=hf_xxx dans le terminal Kaggle, PAS le token lui-même)
    token_env_var: str = "HF_TOKEN"
    push_every_n_transitions: int = 10_000
    local_dir: str = "./hf_checkpoints"


@dataclass
class InfraConfig:
    num_devices: int = 2         # dual-GPU via jax.pmap
    seed: int = 42
    card_csv: str = "/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv"
    # Deck de référence du sample_submission (garanti valide par le moteur)
    # Utilisé pour détecter les cartes Ace Spec et amorcer le replay buffer.
    reference_deck_csv: str = (
        "/kaggle/input/competitions/pokemon-tcg-ai-battle"
        "/sample_submission/sample_submission/deck.csv"
    )
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    debug_no_jit: bool = False   # set True to disable JIT for debugging


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hf: HFConfig = field(default_factory=HFConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)

    # ── Serialisation ─────────────────────────────────────────────────────
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        d = json.loads(s)
        return cls(
            model=ModelConfig(**d["model"]),
            search=SearchConfig(**d["search"]),
            train=TrainConfig(**d["train"]),
            hf=HFConfig(**d["hf"]),
            infra=InfraConfig(**d["infra"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_json(Path(path).read_text())
