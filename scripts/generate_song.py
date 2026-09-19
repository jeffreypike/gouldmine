"""
Agent plays music!
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from gouldmine.midi import array_to_midi, save_midi
from gouldmine.modules import PianoActorCriticMLP


def load_checkpoint(checkpoint_dir, model_class, num_actions=14):
    rngs = nnx.Rngs(0)
    model = model_class(rngs=rngs, num_actions=num_actions)

    _, abstract_state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()

    restored_state = checkpointer.restore(
        os.path.abspath(checkpoint_dir), abstract_state
    )

    nnx.update(model, restored_state)
    return model


def generate_song(key, model, max_steps=320, temperature=0.8, num_actions=14):
    song = jnp.full((max_steps,), num_actions, dtype=jnp.int32)

    for t in range(max_steps):
        time = jnp.array(t)
        logits, _ = model(song, time)

        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits / temperature)
        song = song.at[t].set(action)

    return np.array(song)


if __name__ == "__main__":
    CKPT_PATH = "./wandb/latest-run/files/gouldmine_v1_ckpt"

    model = load_checkpoint(CKPT_PATH, PianoActorCriticMLP)
    seed = 10566
    key = jax.random.PRNGKey(seed)
    song_array = generate_song(key, model, max_steps=320)

    print("Raw output array")
    print(song_array)

    midi = array_to_midi(song_array)
    save_midi(midi, "./sample_song.mid")
