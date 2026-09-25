"""
Different reward engines for teaching piano
"""

import jax
import jax.numpy as jnp

# precomputed KS key matrices
_MAJOR_PROFILE = jnp.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = jnp.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_MAJOR_PROFILE = (_MAJOR_PROFILE - _MAJOR_PROFILE.mean()) / _MAJOR_PROFILE.std()
_MINOR_PROFILE = (_MINOR_PROFILE - _MINOR_PROFILE.mean()) / _MINOR_PROFILE.std()

ALL_KEYS = jnp.concatenate(
    [
        jax.scipy.linalg.circulant(_MAJOR_PROFILE),
        jax.scipy.linalg.circulant(_MINOR_PROFILE),
    ],
    axis=1,
)


def krumhansl_schmuckler(song, action, num_actions):
    """
    Rewards the agent for playing in keys that are more likely to be the key of the song so far.
    """
    note_counts = jnp.bincount(song, length=12)
    note_counts = (note_counts - note_counts.mean()) / (note_counts.std() + 1e-8)

    covariance = jnp.dot(note_counts, ALL_KEYS) / 12
    key_probs = jax.nn.softmax(covariance)
    note_scores = jnp.dot(ALL_KEYS, key_probs)

    safe_action = jnp.minimum(action, num_actions - 3)
    return jnp.where(action < num_actions - 2, note_scores[safe_action], 0)


def sparse_harmonic_reward(
    song,
    action,
    num_actions,
    fatigue_threshold=8,
    diversity_window=8,
    repetition_threshold=3,
):
    """
    Reward harmonic understanding while penalizing the agent for not leaving any space or repeating notes.
    """
    is_padding = song == num_actions
    current_time = jnp.where(jnp.any(is_padding), jnp.argmax(is_padding), song.shape[0])

    ks_reward = krumhansl_schmuckler(song, action, num_actions)

    lookback_window = fatigue_threshold + 8
    lookback_indices = current_time - 1 - jnp.arange(lookback_window)

    safe_indices = jnp.maximum(0, lookback_indices)
    recent_history = song[safe_indices]

    valid_mask = lookback_indices >= 0
    # counts backwards from current time, 0 for rest, 1 otherwise
    is_pitch_recent = (recent_history != num_actions - 2) & valid_mask

    # cumulative product hits zero when it gets to a rest, otherwise 1, so the sum counts notes since last rest
    consecutive_notes = jnp.sum(jnp.cumprod(is_pitch_recent.astype(jnp.int32)))
    rest_not_proposed = action != num_actions - 2
    fatigue_penalty = jnp.where(
        (consecutive_notes >= fatigue_threshold) & rest_not_proposed,
        0.1 * (consecutive_notes - fatigue_threshold + 1),
        0,
    )

    # currently penalizes multiple sustains, even on different notes
    start_idx = jnp.maximum(0, current_time - diversity_window)
    window = jax.lax.dynamic_slice(song, (start_idx,), (diversity_window,))

    appearances = jnp.sum(window == action)
    diversity_penalty = jnp.where(
        (appearances >= repetition_threshold) & (rest_not_proposed), 0.5, 0
    )

    total_reward = ks_reward - fatigue_penalty - diversity_penalty
    return total_reward, {
        "ks": ks_reward,
        "fatigue": fatigue_penalty,
        "diversity": diversity_penalty,
    }
