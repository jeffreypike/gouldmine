"""
Agentic pianists.
"""

import dataclasses

import chex
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


@chex.dataclass
class SongTrajectory:
    song: Array
    rewards: Array
    log_probs: Array
    values: Array
    metrics: dict[str, Array] = dataclasses.field(default_factory=dict)


@chex.dataclass
class PPOMetrics:
    policy_loss: Array
    value_loss: Array
    entropy_loss: Array


class PianoPPOAgent(nnx.Module):
    def __init__(
        self,
        rngs,
        model,
        optimizer,
        reward_fn,
        num_keys=12,
        max_bars=20,
        gamma=0.99,
        lamda=0.95,
        epsilon=0.2,
        c_value=0.5,
        c_entropy=0.01,
    ):
        self.rngs = rngs
        self.num_actions = num_keys + 2  # add rest and sustain
        self.model = model(rngs=rngs, num_actions=self.num_actions)
        self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.Param)
        self.reward_fn = lambda song, action: reward_fn(
            song, action, num_actions=self.num_actions
        )
        self.max_steps = 16 * max_bars
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        self.c_value = c_value
        self.c_entropy = c_entropy

    def sample(self, key, song):
        logits, value = self.model(song)
        action = jax.random.categorical(key, logits)
        full_log_probs = jax.nn.log_softmax(logits)

        log_probs = jnp.take_along_axis(
            full_log_probs, jnp.expand_dims(action, axis=-1), axis=-1
        ).squeeze(axis=-1)
        return action, log_probs, value

    @nnx.jit
    @nnx.vmap(in_axes=(None, 0))
    def collect_trajectories(self, key):
        init_song = jnp.full(self.max_steps, self.num_actions, dtype=jnp.int8)
        init_carry = (init_song, key)

        @nnx.scan
        def step_song(carry, time):
            song, current_key = carry
            current_key, sample_key = jax.random.split(current_key)

            action, log_prob, value = self.sample(sample_key, song)
            total_reward, aux_metrics = self.reward_fn(song, action)

            next_song = song.at[time].set(action)
            next_carry = (next_song, current_key)

            return next_carry, (total_reward, log_prob, value, aux_metrics)

        time_steps = jnp.arange(self.max_steps)
        final_carry, step_outputs = step_song(init_carry, time_steps)
        rewards, log_probs, values, aux_metrics = step_outputs
        final_song = final_carry[0]

        return SongTrajectory(
            song=final_song,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            metrics=aux_metrics,
        )

    @nnx.vmap(in_axes=(None, 0))
    def estimate_advantages(self, trajectory):
        rewards, values = trajectory.rewards, trajectory.values
        chex.assert_rank([rewards, values], 1)

        next_values = jnp.pad(values[1:], (0, 1), mode="constant")
        chex.assert_equal_shape([rewards, values, next_values])

        td_residuals = rewards + self.gamma * next_values - values

        def get_advantages(advantage_tp1, td_t):
            advantage_t = td_t + self.gamma * self.lamda * advantage_tp1
            return advantage_t, advantage_t

        init_advantage = jnp.array(0.0, dtype=jnp.float32)
        _, advantages = jax.lax.scan(
            get_advantages, init_advantage, td_residuals, reverse=True
        )

        target_returns = advantages + values
        return advantages, target_returns

    @chex.assert_max_traces(n=1)
    @chex.chexify
    @nnx.jit
    def update_model(
        self, minibatch_trajectories, minibatch_advantages, minibatch_returns
    ):
        def loss_fn(model, trajectories, advantages, returns):
            def ppo_loss(model, trajectory, advantages, target_returns):
                log_probs = trajectory.log_probs

                chex.assert_equal_shape([log_probs, advantages, target_returns])

                @nnx.vmap(in_axes=(None, 0))
                def causal_forward(song, time):
                    time_indices = jnp.arange(self.max_steps)
                    padding_token = jnp.array(
                        self.num_actions, dtype=jnp.int8
                    )  # using python int will upcast song
                    causal_song = jnp.where(time_indices < time, song, padding_token)
                    logits, value = model(causal_song)
                    return logits, value

                song = trajectory.song
                current_logits, current_values = causal_forward(
                    song, jnp.arange(self.max_steps)
                )

                full_log_probs = jax.nn.log_softmax(current_logits)
                neg_entropies = jnp.sum(
                    jnp.exp(full_log_probs) * full_log_probs, axis=-1
                )

                current_log_probs = jnp.take_along_axis(
                    full_log_probs, jnp.expand_dims(song, axis=-1), axis=-1
                ).squeeze(axis=-1)

                importance_ratio = jnp.exp(current_log_probs - log_probs)
                clipped = jnp.clip(importance_ratio, 1 - self.epsilon, 1 + self.epsilon)

                loss_clipped = (
                    -1
                    * jnp.minimum(
                        importance_ratio * advantages, clipped * advantages
                    ).mean()
                )
                loss_value = ((current_values - target_returns) ** 2).mean()
                loss_entropy = neg_entropies.mean()

                loss_ppo = (
                    loss_clipped
                    + self.c_value * loss_value
                    + self.c_entropy * loss_entropy
                )
                metrics = PPOMetrics(
                    policy_loss=loss_clipped,
                    value_loss=loss_value,
                    entropy_loss=loss_entropy,
                )
                return loss_ppo, metrics

            batch_loss, batch_aux = nnx.vmap(ppo_loss, in_axes=(None, 0, 0, 0))(
                model, trajectories, advantages, returns
            )
            loss = batch_loss.mean()
            aux = jax.tree.map(lambda x: x.mean(), batch_aux)
            return loss, aux

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux_metrics), grads = grad_fn(
            self.model, minibatch_trajectories, minibatch_advantages, minibatch_returns
        )

        chex.assert_tree_all_finite(grads)

        self.optimizer.update(self.model, grads)
        return loss, aux_metrics
