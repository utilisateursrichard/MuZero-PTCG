"""
ptcg_muzero/models/networks.py
================================
The three MuZero networks implemented in Flax / JAX:

  h  – Representation network    : observation → latent state z
  f  – Prediction network        : z → (policy_logits, value)
  g  – Dynamics network          : (z, action) → (reward, z_next)

Architecture
------------
Representation:
  All board entities (active, bench, hand, discard, options…) are embedded as
  card tokens and processed by a Transformer encoder.  The [CLS] output is the
  latent state.

Prediction:
  Two shallow MLPs branching from the latent state.

Dynamics:
  Action is encoded by a learned embedding.  A residual MLP maps
  (z ∥ a_embed) → (z_next, reward).
  *Stochastic transitions* (coin flips) are handled by running the
  dynamics twice (heads / tails) and averaging the outputs
  (collapsed-expectation approximation, cheaper than full Stochastic MuZero).

All three share the same CardEmbedding module whose parameters are part of
the unified parameter tree trained by the learner.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from cards.encoder import CardEmbedding, CARD_STATIC_DIM
from config import ModelConfig
from env.encoding import (
    GLOBAL_FEAT_DIM,
    OPTION_FEAT_DIM,
    POKEMON_FEAT_DIM,
)

# ─────────────────────────────────────────────────────────────────────────────
# Utility blocks
# ─────────────────────────────────────────────────────────────────────────────
class LayerNorm(nn.Module):
    """Standard layer-norm with small epsilon for numerical stability."""
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.LayerNorm(epsilon=1e-5)(x)


class MLP(nn.Module):
    """[hidden → hidden → out] MLP with optional residual connection."""
    hidden_dim: int
    out_dim: int
    use_residual: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.gelu(h)
        h = nn.Dense(self.out_dim)(h)
        if self.use_residual and x.shape[-1] == self.out_dim:
            h = h + x
        return h


class TransformerBlock(nn.Module):
    latent_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        x: [B, T, D]
        mask: [B, T] bool, True where token is VALID (not padding).
        """
        # Self-attention
        attn_mask = None
        if mask is not None:
            # [B, 1, 1, T] → broadcast over heads and query positions
            attn_mask = mask[:, None, None, :]

        h = LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.latent_dim,
            dropout_rate=self.dropout_rate,
        )(h, h, mask=attn_mask, deterministic=deterministic)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        x = x + h

        # Feed-forward
        h = LayerNorm()(x)
        h = nn.Dense(self.ff_dim)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.latent_dim)(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        return x + h


# ─────────────────────────────────────────────────────────────────────────────
# Representation network  h(obs) → z
# ─────────────────────────────────────────────────────────────────────────────
class RepresentationNetwork(nn.Module):
    """
    Encodes the full observation into a latent state.

    Entity sequence (per sample):
      [CLS]  my_active  my_bench×5  my_hand×H  my_discard×D
             opp_active opp_bench×5 opp_discard×D
             option×A
    plus global scalar features injected at every token via concatenation
    then projected.
    """
    cfg: ModelConfig
    static_features: jnp.ndarray   # frozen [num_card_ids, CARD_STATIC_DIM]

    def setup(self) -> None:
        card_total_dim = self.cfg.card_embed_dim + CARD_STATIC_DIM
        D = self.cfg.latent_dim

        self.card_emb = CardEmbedding(
            num_card_ids=self.cfg.num_card_ids,
            embed_dim=self.cfg.card_embed_dim,
            static_features=self.static_features,
        )
        # Project pokemon features (card_token + pokemon_state) to latent_dim
        self.pokemon_proj = nn.Dense(D)
        # Project option features to latent_dim
        self.option_proj  = nn.Dense(D)
        # Global features injected as a single extra token
        self.global_proj  = nn.Dense(D)
        # Learnable CLS token
        self.cls_token    = self.param(
            "cls_token", nn.initializers.normal(0.02), (1, 1, D)
        )
        self.transformer_blocks = [
            TransformerBlock(
                latent_dim=D,
                num_heads=self.cfg.num_heads,
                ff_dim=self.cfg.ff_dim,
                dropout_rate=self.cfg.dropout_rate,
            )
            for _ in range(self.cfg.num_enc_layers)
        ]
        self.out_norm = LayerNorm()

    def __call__(
        self,
        obs: dict,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            obs: batch of encoded observations (each value has leading batch dim B).
        Returns:
            latent_state: [B, latent_dim]
        """
        B = obs["global_feat"].shape[0]
        tokens, masks = self._build_sequence(obs, B)

        # Prepend CLS token (always valid)
        cls  = jnp.broadcast_to(self.cls_token, (B, 1, self.cfg.latent_dim))
        cls_mask = jnp.ones((B, 1), dtype=bool)
        tokens = jnp.concatenate([cls, tokens], axis=1)
        masks  = jnp.concatenate([cls_mask, masks], axis=1)

        x = tokens
        for block in self.transformer_blocks:
            x = block(x, mask=masks, deterministic=deterministic)

        x = self.out_norm(x)
        # Return the CLS token as the latent state
        return x[:, 0, :]   # [B, D]

    def _build_sequence(
        self, obs: dict, B: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Build token sequence and validity mask."""
        D = self.cfg.latent_dim
        tokens: list = []
        masks:  list = []

        def _add_pokemon_tokens(ids, feats, mask_or_one):
            """ids [B,N], feats [B,N,POKEMON_FEAT_DIM], mask [B,N] or scalar."""
            card_tok = self.card_emb(ids)              # [B, N, card_total_dim]
            cat      = jnp.concatenate(
                [card_tok, feats], axis=-1
            )                                          # [B, N, card_total_dim+POKEMON_FEAT_DIM]
            proj = self.pokemon_proj(cat)              # [B, N, D]
            tokens.append(proj)
            if isinstance(mask_or_one, bool) and mask_or_one:
                masks.append(jnp.ones((B, ids.shape[1]), dtype=bool))
            else:
                masks.append(mask_or_one)

        # ── My Pokémon ─────────────────────────────────────────────────────
        _add_pokemon_tokens(
            obs["my_active_id"],   # [B, 1]
            obs["my_active_feat"], # [B, 1, POKEMON_FEAT_DIM]
            True,
        )
        _add_pokemon_tokens(
            obs["my_bench_ids"],   # [B, 5]
            obs["my_bench_feat"],  # [B, 5, POKEMON_FEAT_DIM]
            obs["my_bench_mask"],  # [B, 5]
        )

        # ── My hand (IDs only, no pokemon state) ──────────────────────────
        hand_tok = self.card_emb(obs["my_hand_ids"])    # [B, H, card_dim]
        hand_tok = self.pokemon_proj(
            jnp.concatenate(
                [hand_tok,
                 jnp.zeros((*hand_tok.shape[:-1], POKEMON_FEAT_DIM))],
                axis=-1,
            )
        )
        tokens.append(hand_tok)
        masks.append(obs["my_hand_mask"])

        # ── My discard (IDs only) ─────────────────────────────────────────
        my_dis_tok = self.card_emb(obs["my_discard_ids"])
        my_dis_proj = self.pokemon_proj(
            jnp.concatenate(
                [my_dis_tok,
                 jnp.zeros((*my_dis_tok.shape[:-1], POKEMON_FEAT_DIM))],
                axis=-1,
            )
        )
        tokens.append(my_dis_proj)
        masks.append(obs["my_discard_mask"])

        # ── Opponent Pokémon ───────────────────────────────────────────────
        _add_pokemon_tokens(
            obs["opp_active_id"],
            obs["opp_active_feat"],
            True,
        )
        _add_pokemon_tokens(
            obs["opp_bench_ids"],
            obs["opp_bench_feat"],
            obs["opp_bench_mask"],
        )

        # ── Opp discard (IDs only) ─────────────────────────────────────────
        dis_tok = self.card_emb(obs["opp_discard_ids"])
        dis_proj = self.pokemon_proj(
            jnp.concatenate(
                [dis_tok,
                 jnp.zeros((*dis_tok.shape[:-1], POKEMON_FEAT_DIM))],
                axis=-1,
            )
        )
        tokens.append(dis_proj)
        masks.append(obs["opp_discard_mask"])

        # ── Opponent hand belief (IDs sampled by ISMCTS, mask from hand count) ──
        opp_hand_tok = self.card_emb(obs["opp_hand_ids"])
        opp_hand_proj = self.pokemon_proj(
            jnp.concatenate(
                [opp_hand_tok,
                 jnp.zeros((*opp_hand_tok.shape[:-1], POKEMON_FEAT_DIM))],
                axis=-1,
            )
        )
        tokens.append(opp_hand_proj)
        masks.append(obs["opp_hand_mask"])

        # ── Available options ─────────────────────────────────────────────
        opt_card = self.card_emb(obs["option_ids"])   # [B, A, card_dim]
        opt_feat = obs["option_feat"]                 # [B, A, OPTION_FEAT_DIM]
        opt_tok  = self.option_proj(
            jnp.concatenate([opt_card, opt_feat], axis=-1)
        )
        tokens.append(opt_tok)
        masks.append(obs["option_mask"])

        # ── Global features as one extra token ───────────────────────────
        g_tok = self.global_proj(obs["global_feat"])   # [B, D]
        tokens.append(g_tok[:, None, :])               # [B, 1, D]
        masks.append(jnp.ones((B, 1), dtype=bool))

        return (
            jnp.concatenate(tokens, axis=1),   # [B, T, D]
            jnp.concatenate(masks,  axis=1),   # [B, T]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction network  f(z) → (pi, v)
# ─────────────────────────────────────────────────────────────────────────────
class PredictionNetwork(nn.Module):
    cfg: ModelConfig

    # NOTE (audit §1.0) : un seul `@nn.compact` est autorisé par module Flax.
    # `__call__` délègue donc à `predict_full` sans être lui-même compact.
    def __call__(
        self, z: jnp.ndarray, option_feat: jnp.ndarray | None = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            z: latent state [B, latent_dim]
            option_feat: accepté pour compatibilité d'appel, non utilisé (cf. predict_full)
        Returns:
            policy_logits: [B, max_actions]
            value_scalar:  [B]            (expected scalar in [v_min, v_max])
        """
        pi, v_scalar, _ = self.predict_full(z, option_feat=option_feat)
        return pi, v_scalar

    @nn.compact
    def predict_full(
        self, z: jnp.ndarray, option_feat: jnp.ndarray | None = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            policy_logits: [B, max_actions]
            value_scalar:  [B]            (expected scalar in [v_min, v_max])
            value_logits:  [B, num_bins]   (categorical distribution logits)

        AUDIT §1.1 — la branche « Pointer Network » (pi_q / pi_k) a été supprimée.
        Elle n'était active que lorsque `option_feat` était fourni, c'est-à-dire
        uniquement à la racine de la self-play / inférence, JAMAIS dans la loss ni
        dans le `recurrent_fn` de mctx.  Ses poids ne recevaient donc aucun
        gradient et injectaient un bruit figé d'amplitude O(1) dans la policy
        prior, réappris en boucle par le réseau → policy collapse.
        La tête est maintenant strictement identique à la racine et dans l'arbre.
        `option_feat` reste accepté (et ignoré) pour ne pas casser les appelants.
        """
        del option_feat  # volontairement inutilisé — cf. docstring
        D = self.cfg.latent_dim
        num_bins = getattr(self.cfg, "num_value_bins", 51)
        v_min = getattr(self.cfg, "value_min", -1.8)
        v_max = getattr(self.cfg, "value_max", 1.8)

        # Policy head
        pi_hid = MLP(hidden_dim=D, out_dim=D, use_residual=False)(z)
        pi_hid = LayerNorm()(pi_hid)

        pi_logits = nn.Dense(self.cfg.max_actions, name="pi_dense")(pi_hid)

        # Value head (Categorical logits)
        v_hid = MLP(hidden_dim=D, out_dim=D, use_residual=False)(z)
        v_hid = LayerNorm()(v_hid)
        v_logits = nn.Dense(num_bins, name="v_dense")(v_hid)

        # Decode expected value scalar for MCTS
        v_probs = jax.nn.softmax(v_logits, axis=-1)
        bins = jnp.linspace(v_min, v_max, num_bins)
        v_scalar = jnp.sum(v_probs * bins, axis=-1)

        return pi_logits, v_scalar, v_logits


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics network  g(z, a) → (r, z_next)
# ─────────────────────────────────────────────────────────────────────────────
class DynamicsNetwork(nn.Module):
    """
    Simulates one game step in latent space.
    Receives either action_feat (45-dim semantic vector) or action_onehot.
    """
    cfg: ModelConfig

    # NOTE (audit §1.0) : un seul `@nn.compact` par module Flax.
    def __call__(
        self,
        z: jnp.ndarray,
        action_onehot: jnp.ndarray,
        action_feat: jnp.ndarray | None = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            z:             [B, latent_dim]
            action_onehot: [B, max_actions]
            action_feat:   accepté pour compatibilité, non utilisé
        Returns:
            reward:  [B] (expected scalar)
            z_next:  [B, latent_dim]
        """
        r_scalar, _, z_next = self.dynamics_full(z, action_onehot, action_feat=action_feat)
        return r_scalar, z_next

    @nn.compact
    def dynamics_full(
        self,
        z: jnp.ndarray,
        action_onehot: jnp.ndarray,
        action_feat: jnp.ndarray | None = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            reward_scalar: [B]            (expected scalar)
            reward_logits: [B, num_bins]   (categorical reward logits)
            z_next:        [B, latent_dim]

        AUDIT §1.1 — `action_feat` n'était fourni par aucun appelant : la couche
        `a_feat_emb` était du poids mort.  Supprimée ; seul l'embedding one-hot
        est conservé, ce qui rend la dynamique identique partout.
        """
        del action_feat  # volontairement inutilisé — cf. docstring
        D = self.cfg.latent_dim
        num_bins = getattr(self.cfg, "num_value_bins", 51)
        v_min = getattr(self.cfg, "value_min", -1.8)
        v_max = getattr(self.cfg, "value_max", 1.8)

        a_emb = nn.Dense(D, name="a_emb")(action_onehot)
        a_emb = nn.gelu(a_emb)

        x = jnp.concatenate([z, a_emb], axis=-1)   # [B, 2D]

        # Clean deterministic transition (prevents linear vector averaging / OOD states)
        z_next = self._transition_mlp(x, name="det")
        r_logits = self._reward_mlp(x, name="rdet", num_bins=num_bins)

        r_probs = jax.nn.softmax(r_logits, axis=-1)
        bins = jnp.linspace(v_min, v_max, num_bins)
        r_scalar = jnp.sum(r_probs * bins, axis=-1)

        return r_scalar, r_logits, z_next


    def _transition_mlp(
        self, x: jnp.ndarray, *, name: str
    ) -> jnp.ndarray:
        D = self.cfg.latent_dim
        h = nn.Dense(D * 2, name=f"{name}_fc1")(x)
        h = nn.gelu(h)
        h = nn.Dense(D, name=f"{name}_fc2")(h)
        h = LayerNorm()(h)
        return h

    def _reward_mlp(
        self, x: jnp.ndarray, *, name: str, num_bins: int = 51
    ) -> jnp.ndarray:
        D = self.cfg.latent_dim
        h = nn.Dense(D // 2, name=f"{name}_fc1")(x)
        h = nn.gelu(h)
        return nn.Dense(num_bins, name=f"{name}_fc2")(h)



# ─────────────────────────────────────────────────────────────────────────────
# Projector and Predictor for EfficientZero Consistency Loss
# ─────────────────────────────────────────────────────────────────────────────
class Projector(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        D = self.cfg.latent_dim
        h = nn.Dense(D, name="fc1")(x)
        h = nn.gelu(h)
        h = nn.Dense(D, name="fc2")(h)
        return h


class Predictor(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        D = self.cfg.latent_dim
        h = nn.Dense(D, name="fc1")(x)
        h = nn.gelu(h)
        h = nn.Dense(D, name="fc2")(h)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Unified MuZero model (all three networks + projector/predictor)
# ─────────────────────────────────────────────────────────────────────────────
class MuZeroNetwork(nn.Module):
    """
    Combines h + f + g into a single Flax module whose parameter tree is
    managed by the trainer.
    """
    cfg: ModelConfig
    static_features: jnp.ndarray   # frozen card feature matrix

    def setup(self) -> None:
        self.h = RepresentationNetwork(
            cfg=self.cfg, static_features=self.static_features
        )
        self.f = PredictionNetwork(cfg=self.cfg)
        self.g = DynamicsNetwork(cfg=self.cfg)
        self.project = Projector(cfg=self.cfg)
        self.predict_next = Predictor(cfg=self.cfg)

    def represent(
        self, obs: dict, deterministic: bool = True
    ) -> jnp.ndarray:
        return self.h(obs, deterministic=deterministic)

    def predict(
        self, z: jnp.ndarray, option_feat: jnp.ndarray | None = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.f(z, option_feat=option_feat)

    def predict_full(
        self, z: jnp.ndarray, option_feat: jnp.ndarray | None = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.f.predict_full(z, option_feat=option_feat)

    def dynamics(
        self, z: jnp.ndarray, action_onehot: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r_scalar, z_next = self.g(z, action_onehot)
        return z_next, r_scalar

    def dynamics_full(
        self, z: jnp.ndarray, action_onehot: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.g.dynamics_full(z, action_onehot)

    def project_state(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.project(z)

    def predict_state(self, proj_z: jnp.ndarray) -> jnp.ndarray:
        return self.predict_next(proj_z)

    def __call__(
        self, obs: dict, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Full forward pass for the root node: h → f."""
        z        = self.represent(obs, deterministic=deterministic)
        opt_feat = obs.get("option_feat", None) if isinstance(obs, dict) else None
        pi, v    = self.predict(z, option_feat=opt_feat)
        return z, pi, v

    def init_all(
        self, obs: dict, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Eagerly initializes all submodules (h, f, g, project, predict_next) during init."""
        z        = self.represent(obs, deterministic=deterministic)
        opt_feat = obs.get("option_feat", None) if isinstance(obs, dict) else None
        pi, v    = self.predict(z, option_feat=opt_feat)

        dummy_action = jnp.zeros((z.shape[0], self.cfg.max_actions))
        _, _ = self.dynamics(z, dummy_action)
        
        proj_z = self.project_state(z)
        _ = self.predict_state(proj_z)
        
        return z, pi, v
