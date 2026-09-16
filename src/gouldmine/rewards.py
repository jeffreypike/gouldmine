"""
Different reward engines for teaching piano
"""

import jax.numpy as jnp


def dummy_reward_fn(song, action):
    target_token = 0

    return jnp.where(action == target_token, 1.0, -0.1)
