"""
Training run.
"""

import dataclasses
import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
from flax import nnx
from tqdm.rich import trange

import wandb
from gouldmine.agents import PianoPPOAgent
from gouldmine.modules import PianoActorCriticMLP
from gouldmine.rewards import sparse_harmonic_reward


@dataclass
class EnvConfig:
    num_keys = 12
    max_bars = 20


@dataclass
class PPOConfig:
    gamma = 0.99
    lamda = 0.95
    epsilon = 0.2
    c_value = 0.5
    c_entropy = 0.01


@dataclass
class TrainConfig:
    num_updates = 1000
    num_epochs = 4
    batch_size = 128
    minibatch_size = 32
    learning_rate = 0.001

    @property
    def num_minibatches(self) -> int:
        return self.batch_size // self.minibatch_size


@dataclass
class RunConfig:
    seed: int = 10566
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)


if __name__ == "__main__":
    config = tyro.cli(RunConfig)

    wandb.init(project="gouldmine", config=dataclasses.asdict(config))

    key = jax.random.PRNGKey(seed=config.seed)
    rngs: nnx.Rngs = nnx.Rngs(params=config.seed)

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=config.train.learning_rate),
    )

    agent = PianoPPOAgent(
        rngs=rngs,
        model=PianoActorCriticMLP,
        optimizer=optimizer,
        reward_fn=sparse_harmonic_reward,
        num_keys=config.env.num_keys,
        max_bars=config.env.max_bars,
        gamma=config.ppo.gamma,
        lamda=config.ppo.lamda,
        epsilon=config.ppo.epsilon,
        c_value=config.ppo.c_value,
        c_entropy=config.ppo.c_entropy,
    )

    pbar = trange(config.train.num_updates, desc="Training Agent")
    for j in pbar:
        start_time = time.time()
        traj_key, key = jax.random.split(key)
        traj_keys = jax.random.split(traj_key, config.train.batch_size)
        trajectories = agent.collect_trajectories(traj_keys)  # pyright: ignore[reportCallIssue]
        advantages, returns = agent.estimate_advantages(trajectories)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        mean_reward = trajectories.rewards.sum(axis=1).mean()
        epoch_losses = []

        for _ in range(config.train.num_epochs):
            shuffle_key, key = jax.random.split(key)
            permutation = jax.random.permutation(shuffle_key, config.train.batch_size)

            shuffled_trajectories = jax.tree.map(
                lambda x, p=permutation: x[p], trajectories
            )
            shuffled_advantages = advantages[permutation, :]
            shuffled_returns = returns[permutation, :]

            batched_trajectories = jax.tree.map(
                lambda x: x.reshape(
                    (config.train.num_minibatches, config.train.minibatch_size, -1)
                ),
                shuffled_trajectories,
            )
            batched_advantages = shuffled_advantages.reshape(
                (config.train.num_minibatches, config.train.minibatch_size, -1)
            )
            batched_returns = shuffled_returns.reshape(
                (config.train.num_minibatches, config.train.minibatch_size, -1)
            )

            for i in range(config.train.num_minibatches):
                minibatch_trajectories = jax.tree.map(
                    lambda x, idx=i: x[idx], batched_trajectories
                )
                minibatch_advantages = batched_advantages[i]
                minibatch_returns = batched_returns[i]

                loss, aux_metrics = agent.update_model(
                    minibatch_trajectories,  # pyright: ignore[reportCallIssue]
                    minibatch_advantages,
                    minibatch_returns,
                )
                epoch_losses.append(
                    [
                        loss,
                        aux_metrics.policy_loss,
                        aux_metrics.value_loss,
                        aux_metrics.entropy_loss,
                    ]
                )

        mean_losses = np.mean(epoch_losses, axis=0)
        step_duration = time.time() - start_time
        total_env_steps = config.train.batch_size * agent.max_steps
        steps_per_second = total_env_steps / step_duration
        mean_reward_metrics = jax.tree.map(jnp.mean, trajectories.metrics)

        wandb.log(
            {
                # "update": j,
                "rollout/mean_reward": mean_reward.item(),
                "reward/ks_score": mean_reward_metrics.get("ks", jnp.array(0.0)).item(),
                "reward/fatigue_penalty": mean_reward_metrics.get(
                    "fatigue", jnp.array(0.0)
                ).item(),
                "reward/diversity_penalty": mean_reward_metrics.get(
                    "divergence", jnp.array(0.0)
                ).item(),
                "train/loss_total": mean_losses[0].item(),
                "train/loss_policy": mean_losses[1].item(),
                "train/loss_value": mean_losses[2].item(),
                "train/loss_entropy": mean_losses[3].item(),
                "time/step_duration_sec": step_duration,
                "time/steps_per_second": steps_per_second,
            }
        )
        pbar.set_postfix(
            reward=f"{mean_reward.item():.2f}",
            entropy=f"{mean_losses[3].item():.4f}",
            val_loss=f"{mean_losses[2].item():.4f}",
            sps=f"{int(steps_per_second)}",
        )

    print("Saving Gouldmine checkpoint...")
    graphdef, state = nnx.split(agent.model)

    if wandb.run is not None and wandb.run.dir is not None:
        base_dir = wandb.run.dir
    else:
        base_dir = os.getcwd()
    checkpoint_dir = os.path.join(base_dir, "gouldmine_v1_ckpt")
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.abspath(checkpoint_dir), state, force=True)

    print("Generating sample song...")
    sample_song = jnp.full((agent.max_steps,), agent.num_actions, dtype=jnp.int8)
    for t in range(agent.max_steps):
        time = jnp.array(t)
        logits, _ = agent.model(sample_song, time)
        logits = logits[t]
        next_action = jnp.argmax(logits)
        sample_song = sample_song.at[t].set(next_action)
    sample_song_np = np.array(sample_song)
    sample_path = os.path.join(base_dir, "sample_output.npy")
    np.save(sample_path, sample_song_np)
    print(f"Sample saved to {sample_path}")
    print("Final Song Output:")
    print(sample_song_np)

    wandb.finish()
