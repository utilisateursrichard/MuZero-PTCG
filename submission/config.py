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
# AUDIT §3.5 : la valeur faisant autorité est celle de cards/encoder.py (= 6).
# Basic, Stage1, Stage2, BasicEnergy, SpecialEnergy, Trainer/inconnu.
NUM_STAGES: int = 6
# CARD_STATIC_DIM is defined in cards/encoder.py and equals 48


@dataclass
class ModelConfig:
    # ── Card representation ────────────────────────────────────────────────
    num_card_ids: int = 1268     # safe upper-bound; max(CSV card_id) + 1
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
    num_probe_tasks: int = 11

    # ── Deck builder ──────────────────────────────────────────────────────
    num_deck_slots: int = 60

    # ── Value & Reward Categorical Bins ────────────────────────────────────
    num_value_bins: int = 51
    value_min: float = -1.8
    value_max: float = 1.8



@dataclass
class SearchConfig:
    num_simulations: int = 50           # 50 simulations ISMCTS
    # ISMCTS: number of determinizations sampled at each root
    num_belief_samples: int = 4

    # Gumbel MuZero: number of considered actions at each node
    max_num_considered_actions: int = 8  # réduit de 16 → 8 : sélectionne les meilleures actions
    # Dirichlet noise at root (self-play exploration)
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    # Temperature scheduling for self-play action selection
    temperature_init: float = 0.8        # Température initiale (tours de setup 1-2)
    temperature_min: float = 0.15        # Température minimale en milieu/fin de partie (maintient l'exploration résiduelle)
    temperature_decay: float = 0.85      # Décroissance par tour : tau(turn) = max(tau_min, tau_init * decay^(turn-1))
    # AUDIT §1.2 — le compteur de tours doit provenir du moteur (`current.turn`).
    # Si le moteur ne l'expose pas, on compte les alternances de joueur, JAMAIS
    # le nombre d'étapes moteur (qui inclut chaque sous-décision).
    temperature_min_turn_estimate: bool = True

    # AUDIT §2.1 — échantillonner la main adverse dans le deck réellement joué
    # plutôt que dans les 1268 IDs du pool complet.
    belief_from_known_deck: bool = True


@dataclass
class TrainConfig:
    # ── MuZero unrolling ──────────────────────────────────────────────────
    num_unroll_steps: int = 5
    td_steps: int = 20
    gamma: float = 1.0
    target_network_tau: float = 0.995   # EMA decay factor for Target Network Polyak averaging

    # ── Optimiser ─────────────────────────────────────────────────────────
    # 256 samples required ~19.5 GB per GPU with the current transformer and
    # MuZero unroll.  64 keeps each of the two replicas at 32 samples.
    batch_size: int = 64         # total across all devices
    learning_rate: float = 3e-4
    lr_warmup_steps: int = 2_000
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # ── Replay buffer ─────────────────────────────────────────────────────
    replay_buffer_size: int = 200_000
    min_replay_size: int = 10_000
    replay_alpha: float = 0.5    # priority exponent (0 = uniform, 1 = full priority)
    replay_beta: float = 0.4     # IS correction exponent (annealed toward 1 during training)

    # ── Early Plateau unfreezing (h(s)) ───────────────────────────────────
    plateau_window: int = 500
    plateau_threshold: float = 0.005

    # ── Training schedule ─────────────────────────────────────────────────
    num_workers: int = 0             # 0 = auto-scale basé sur os.cpu_count() (jusqu'à 64)
    num_total_steps: int = 500_000
    self_play_interval: int = 100    # global steps between self-play batches
    games_per_self_play: int = 8

    # Safety bound for a single engine episode.  A normal game is far shorter;
    # this prevents a non-progressing engine/action loop from stalling training.
    max_game_steps: int = 2_000

    checkpoint_every: int = 1_000
    buffer_push_every: int = 3_000   # async replay buffer push to HF Hub
    eval_every: int = 5_000
    eval_games: int = 20

    # ── Reanalyze (In-Pipeline GPU) ────────────────────────────────────────
    reanalyze_num_simulations: int = 25   # Budget complet pour Sequential Halving (vs 10 précédemment)

    # ── Priority Refresh ───────────────────────────────────────────────────
    priority_refresh_every: int = 500     # steps between refreshes
    priority_refresh_fraction: float = 0.05  # fraction considered per refresh
    priority_refresh_max_entries: int = 256  # hard bound: attention is O(batch²)
    priority_refresh_batch_size: int = 32    # GPU micro-batch for the forward pass

    # ── Loss weights ──────────────────────────────────────────────────────
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    policy_entropy_weight: float = 0.02   # Bonus d'entropie étendu pour maintenir l'exploration active
    probe_loss_weight: float = 0.05
    deck_loss_weight: float = 0.1
    consistency_loss_weight: float = 2.0

    # ── Reward Shaping ────────────────────────────────────────────────────
    enable_reward_shaping: bool = True
    prize_reward: float = 1.0 / 12.0     # ~ +0.0833 par prize prise (+0.50 pour 6 prizes)
    prize_penalty: float = 1.0 / 12.0    # ~ -0.0833 par prize concédée (-0.50 pour 6 prizes)
    win_reward: float = 1.0              # Récompense terminale en cas de victoire
    loss_penalty: float = 1.0            # Pénalité terminale en cas de défaite
    deck_out_penalty: float = 0.1        # Malus additionnel si défaite par épuisement du deck

    # ── Hot-fix Reset & Dégel Progressif de h(s) ──────────────────────────
    hot_fix: bool = False                 # Active le reset chirurgical complet (f, g, Adam, fresh buffer)
    reset_policy_head: bool = False       # Réinitialise uniquement la tête de prédiction f (policy + value)
    reset_value_head: bool = False        # Réinit. chirurgicale de v_dense + rdet_fc2 UNIQUEMENT
    reset_dynamics_head: bool = False    # Réinitialise la tête de dynamique g (transitions + 50D action emb)
    fresh_buffer: bool = False            # Démarre avec un buffer vide (ignore les anciennes parties passives)
    freeze_representation_steps: int = 2000  # Nombre de steps avec h(s) gelé à 100% (gradient scale 0.0)
    unfreeze_ramp_steps: int = 5000          # Steps de transition pour dégel progressif (gradient scale 0.0 -> 1.0)
    resume_step: Optional[int] = None        # Étape précise à charger (HF ou local)
    resume_ckpt: Optional[str] = None        # Chemin précis d'un checkpoint à charger

    # ── Deck builder ──────────────────────────────────────────────────────
    # AUDIT §2.4 — tant que la self-play utilise DEFAULT_COMPETITIVE_DECK (deck
    # figé, identique pour les deux joueurs), REINFORCE reçoit un `deck_ids`
    # constant et des récompenses opposées : les gradients ne portent aucune
    # information et font dériver les logits.  Désactivé par défaut.
    deck_builder_enabled: bool = False
    deck_lr: float = 1e-3
    deck_entropy_coef: float = 0.01
    deck_baseline_ema: float = 0.99    # exponential moving average for REINFORCE baseline


@dataclass
class HFConfig:
    enabled: bool = True
    repo_id: str = "richard151111/muzero-V2"
    private: bool = True
    # Nom de la variable d'environnement contenant le token HF
    # (ex: export HF_TOKEN=hf_xxx dans le terminal Kaggle, PAS le token lui-même)
    token_env_var: str = "HF_TOKEN"
    push_every_n_transitions: int = 10_000
    local_dir: str = "./hf_checkpoints"


@dataclass
class WandBConfig:
    enabled: bool = True
    project: str = "muzero"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"   # "online", "offline", or "disabled"
    token_env_var: str = "WANDB_API_KEY"
    kaggle_secret_name: str = "WANDB"


@dataclass
class InfraConfig:
    num_devices: int = 2         # dual-GPU via jax.pmap
    num_learner_devices: int = 0 # 0 = auto 50/50 (ex: 4 learners / 4 actors sur 8 TPUs)
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
    wandb: WandBConfig = field(default_factory=WandBConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)

    # ── Serialisation ─────────────────────────────────────────────────────
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        d = json.loads(s)
        def _filter_dc(target_cls, data):
            if not isinstance(data, dict):
                return target_cls()
            import dataclasses
            valid = {f.name for f in dataclasses.fields(target_cls)}
            return target_cls(**{k: v for k, v in data.items() if k in valid})

        return cls(
            model=_filter_dc(ModelConfig, d.get("model", {})),
            search=_filter_dc(SearchConfig, d.get("search", {})),
            train=_filter_dc(TrainConfig, d.get("train", {})),
            hf=_filter_dc(HFConfig, d.get("hf", {})),
            wandb=_filter_dc(WandBConfig, d.get("wandb", {})),
            infra=_filter_dc(InfraConfig, d.get("infra", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_json(Path(path).read_text())
