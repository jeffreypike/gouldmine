"""
Deep learning modules
"""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float32, Int32


class PianoActorCriticMLP(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_actions: int = 14,
        max_bars: int = 20,
        embed_dim: int = 32,
        n_hidden: int = 3,
        d_hidden: int = 128,
    ):
        self.num_actions = num_actions
        self.song_length = max_bars * 16

        self.song_encoding = nnx.Embed(
            num_embeddings=self.num_actions + 1, features=embed_dim, rngs=rngs
        )

        self.input = nnx.Linear(embed_dim * self.song_length + 1, d_hidden, rngs=rngs)
        self.hidden_layers = nnx.List(
            [nnx.Linear(d_hidden, d_hidden, rngs=rngs) for _ in range(n_hidden)]
        )
        self.policy = nnx.Linear(d_hidden, self.num_actions, rngs=rngs)
        self.value = nnx.Linear(d_hidden, 1, rngs=rngs)

    def __call__(
        self, song: Int32[Array, "... song_length"], time: Int32[Array, "..."]
    ) -> tuple[Float32[Array, "... num_actions"], Float32[Array, "..."]]:
        song_encoding = self.song_encoding(song)
        batch_shape = song_encoding.shape[:-2]
        song_encoding = song_encoding.reshape(
            batch_shape + (-1,)
        )  # flatten song_length x embed_dim

        time_rescaled = time.astype(jnp.float32) / self.song_length
        time_rescaled = jnp.expand_dims(time_rescaled, -1)

        x = jnp.concat([song_encoding, time_rescaled], axis=-1)
        x = self.input(x)
        x = nnx.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = nnx.relu(x)
        return self.policy(x), self.value(x).squeeze(axis=-1)
