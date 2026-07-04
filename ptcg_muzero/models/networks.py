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

    @nn.compact
    def __call__(
        self, z: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            z: latent state [B, latent_dim]
        Returns:
            policy_logits: [B, max_actions]
            value:         [B]            (scalar in [-1, 1] via tanh)
        """
        D = self.cfg.latent_dim

        # Policy head
        pi = MLP(hidden_dim=D, out_dim=D, use_residual=False)(z)
        pi = LayerNorm()(pi)
        pi = nn.Dense(self.cfg.max_actions)(pi)

        # Value head
        v  = MLP(hidden_dim=D, out_dim=D, use_residual=False)(z)
        v  = LayerNorm()(v)
        v  = nn.Dense(1)(v)
        v  = jnp.tanh(v)[..., 0]    # [B]

        return pi, v


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics network  g(z, a) → (r, z_next)
# ─────────────────────────────────────────────────────────────────────────────
class DynamicsNetwork(nn.Module):
    """
    Simulates one game step in latent space.

    Action is a one-hot over max_actions.
    The network also outputs an `is_stochastic` logit; if high, a collapsed
    expectation over 2 chance outcomes (coin flip heads/tails) is computed:

        z_next = 0.5 * z_heads + 0.5 * z_tails

    This is the "collapsed chance node" approximation.
    """
    cfg: ModelConfig

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        action_onehot: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            z:             [B, latent_dim]
            action_onehot: [B, max_actions]
        Returns:
            reward:  [B]
            z_next:  [B, latent_dim]
        """
        D = self.cfg.latent_dim

        # Encode action
        a_emb = nn.Dense(D)(action_onehot)  # [B, D]
        a_emb = nn.gelu(a_emb)

        x = jnp.concatenate([z, a_emb], axis=-1)   # [B, 2D]

        # Is this transition stochastic (coin-flip)?
        stoch_logit = nn.Dense(1)(x)               # [B, 1]
        stoch_prob  = jax.nn.sigmoid(stoch_logit[..., 0])  # [B]

        # Deterministic branch
        z_det  = self._transition_mlp(x, name="det")
        r_det  = self._reward_mlp(x, name="rdet")

        # Heads branch (coin = heads → typically better outcome)
        x_h   = jnp.concatenate([x, jnp.ones_like(a_emb[:, :1])], axis=-1)
        z_h   = self._transition_mlp(x_h, name="heads")
        r_h   = self._reward_mlp(x_h, name="rheads")

        # Tails branch
        x_t   = jnp.concatenate([x, -jnp.ones_like(a_emb[:, :1])], axis=-1)
        z_t   = self._transition_mlp(x_t, name="tails")
        r_t   = self._reward_mlp(x_t, name="rtails")

        # Collapse: expected next state weighted by stochasticity probability
        z_coin = 0.5 * z_h + 0.5 * z_t     # expected over coin flip
        r_coin = 0.5 * r_h + 0.5 * r_t

        p = stoch_prob[:, None]
        z_next = (1.0 - p) * z_det + p * z_coin
        reward = (1.0 - stoch_prob) * r_det + stoch_prob * r_coin

        return reward, z_next

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
        self, x: jnp.ndarray, *, name: str
    ) -> jnp.ndarray:
        D = self.cfg.latent_dim
        h = nn.Dense(D // 2, name=f"{name}_fc1")(x)
        h = nn.gelu(h)
        h = nn.Dense(1, name=f"{name}_fc2")(h)
        return jnp.tanh(h)[..., 0]   # scalar per sample


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
        self, z: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.f(z)

    def dynamics(
        self, z: jnp.ndarray, action_onehot: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.g(z, action_onehot)

    def project_state(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.project(z)

    def predict_state(self, proj_z: jnp.ndarray) -> jnp.ndarray:
        return self.predict_next(proj_z)

    def __call__(
        self, obs: dict, deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Full forward pass for the root node: h → f."""
        z        = self.represent(obs, deterministic=deterministic)
        pi, v    = self.predict(z)
        
        # Force l'initialisation des paramètres du modèle de dynamique (self.g)
        # et des têtes de projection/prédiction pendant l'init sans affecter la sortie de la racine.
        dummy_action = jnp.zeros((z.shape[0], self.cfg.max_actions))
        _, _ = self.dynamics(z, dummy_action)
        
        proj_z = self.project_state(z)
        _ = self.predict_state(proj_z)
        
        return z, pi, v
