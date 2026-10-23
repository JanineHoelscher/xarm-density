"""Smooth obstacle window, included locally from the original density backend."""

import jax.numpy as jnp


def smoothstep_cinfty(z):
    """Zero for z <= 0, one for z >= 1, and a smooth transition in between."""
    clipped = jnp.clip(z, 1e-8, 1 - 1e-8)
    left = jnp.exp(-1 / clipped)
    right = jnp.exp(-1 / (1 - clipped))
    interior = left / jnp.maximum(left + right, 1e-300)
    return jnp.where(z <= 0, 0., jnp.where(z >= 1, 1., interior))
