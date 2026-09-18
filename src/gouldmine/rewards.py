"""
Different reward engines for teaching piano
"""

import jax
import jax.numpy as jnp


def krumhansl_schmuckler(song, action, num_actions):
    """
    Rewards the agent for playing in keys that are more likely to be the key of the song so far.
    """
    major_profile = jnp.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = jnp.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    major_profile = (major_profile - major_profile.mean()) / major_profile.std()
    minor_profile = (minor_profile - minor_profile.mean()) / minor_profile.std()

    def convert_sustains(song, num_actions):
        def scan_fn(carry, x):
            val = jnp.where(x == num_actions - 1, carry, x)
            return val, val

        init_carry = song[0]
        return jax.lax.scan(scan_fn, init_carry, song)

    # note: only works for single octave atm, gotta use modular arithmetic when there's more
    _, converted_song = convert_sustains(song, num_actions)
    note_counts = jnp.bincount(converted_song, length=12)
    note_counts = (note_counts - note_counts.mean()) / (note_counts.std() + 1e-8)

    major_keys = jax.scipy.linalg.circulant(major_profile)
    minor_keys = jax.scipy.linalg.circulant(minor_profile)
    all_keys = jnp.concatenate([major_keys, minor_keys], axis=1)

    covariance = jnp.dot(note_counts, all_keys) / 12
    key_probs = jax.nn.softmax(covariance)
    note_scores = jnp.dot(all_keys, key_probs)

    safe_action = jnp.minimum(action, num_actions - 3)
    clamped_reward = note_scores[safe_action]
    reward = jnp.where(action < num_actions - 2, clamped_reward, 0)
    return reward


def sparse_harmonic_reward(
    song,
    action,
    num_actions,
    fatigue_window=8,
    diversity_window=8,
    repetition_threshold=3,
):
    """
    Reward harmonic compatibility while penalizing the agent for not leaving any space or repeating notes.
    """
    is_padding = jnp.argmax(song == num_actions)
    current_time = jnp.where(jnp.any(is_padding), jnp.argmax(is_padding), song.shape[0])

    ks_reward = krumhansl_schmuckler(song, action, num_actions)

    is_pitch = song != num_actions - 2  # all non-rests

    def scan_fatigue(consecutive_note_count, pitch_played):
        "Count consecutive notes without taking a rest"
        next_count = jnp.where(pitch_played, consecutive_note_count + 1, 0)
        return next_count, next_count

    rest_not_proposed = action != num_actions - 2
    _, fatigue_array = jax.lax.scan(scan_fatigue, 0, is_pitch)
    consecutive_notes = jnp.where(current_time > 0, fatigue_array[current_time - 1], 0)
    fatigue_penalty = jnp.where(
        (consecutive_notes >= fatigue_window) & rest_not_proposed,
        0.1 * (consecutive_notes - fatigue_window + 1),
        0,
    )

    # currently penalizes multiple sustains, even on different notes
    start_idx = jnp.maximum(0, current_time - diversity_window)
    window = jax.lax.dynamic_slice(song, (start_idx,), (diversity_window,))
    appearances = jnp.sum(window == action)
    repetitive_note = (appearances >= repetition_threshold) & (rest_not_proposed)
    diversity_penalty = 0.5 * repetitive_note

    total_reward = ks_reward - fatigue_penalty - diversity_penalty
    return total_reward, (ks_reward, fatigue_penalty, diversity_penalty)
