"""
Training run.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm.rich import trange

import wandb
from gouldmine.agents import PianoPPOAgent
from gouldmine.modules import PianoActorCriticMLP
from gouldmine.rewards import dummy_reward_fn

num_updates = 1000
num_epochs = 4
batch_size = 128
minibatch_size = 32
learning_rate = 0.001
num_minibatches = batch_size // minibatch_size

seed = 10566
key = jax.random.PRNGKey(seed=seed)
rngs = nnx.Rngs(params=seed)

wandb.init(
    project="gouldmine",
    config={
        "num_updates": num_updates,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    },
)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5), optax.adam(learning_rate=learning_rate)
)

agent = PianoPPOAgent(
    rngs=rngs,
    model=PianoActorCriticMLP,
    optimizer=optimizer,
    reward_fn=dummy_reward_fn,
)

pbar = trange(num_updates, desc="Training Agent")
for j in pbar:
    start_time = time.time()
    traj_key, key = jax.random.split(key)
    traj_keys = jax.random.split(traj_key, batch_size)
    trajectories = agent.collect_trajectories(traj_keys)
    advantages, returns = agent.estimate_advantages(trajectories)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    mean_reward = trajectories.rewards.sum(axis=1).mean()
    epoch_losses = []

    for _ in range(num_epochs):
        shuffle_key, key = jax.random.split(key)
        permutation = jax.random.permutation(shuffle_key, batch_size)

        shuffled_trajectories = jax.tree.map(
            lambda x, p=permutation: x[p], trajectories
        )
        shuffled_advantages = advantages[permutation, :]
        shuffled_returns = returns[permutation, :]

        batched_trajectories = jax.tree.map(
            lambda x: x.reshape((num_minibatches, minibatch_size, -1)),
            shuffled_trajectories,
        )
        batched_advantages = shuffled_advantages.reshape(
            (num_minibatches, minibatch_size, -1)
        )
        batched_returns = shuffled_returns.reshape(
            (num_minibatches, minibatch_size, -1)
        )

        for i in range(num_minibatches):
            minibatch_trajectories = jax.tree.map(
                lambda x, idx=i: x[idx], batched_trajectories
            )
            minibatch_advantages = batched_advantages[i]
            minibatch_returns = batched_returns[i]

            loss, aux_metrics = agent.update_model(
                minibatch_trajectories, minibatch_advantages, minibatch_returns
            )
            epoch_losses.append((loss, *aux_metrics))

    mean_losses = np.mean(epoch_losses, axis=0)
    step_duration = time.time() - start_time
    total_env_steps = batch_size * agent.max_steps
    steps_per_second = total_env_steps / step_duration
    mean_ks, mean_fatigue, mean_div = jax.tree.map(jnp.mean, trajectories.metrics)

    wandb.log(
        {
            # "update": j,
            "rollout/mean_reward": mean_reward.item(),
            "reward/ks_score": mean_ks.item(),
            "reward/fatigue_penalty": mean_fatigue.item(),
            "reward/diversity_penalty": mean_div.item(),
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

checkpoint_dir = os.path.join(wandb.run.dir, "gouldmine_v1_ckpt")
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(os.path.abspath(checkpoint_dir), state, force=True)

print("Generating sample song...")
sample_song = jnp.full((agent.max_steps,), agent.num_actions, dtype=jnp.int32)
for t in range(agent.max_steps):
    logits, _ = agent.model(sample_song, t)
    logits = logits[t]
    next_action = jnp.argmax(logits)
    sample_song = sample_song.at[t].set(next_action)
sample_song_np = np.array(sample_song)
sample_path = os.path.join(wandb.run.dir, "sample_output.npy")
np.save(sample_path, sample_song_np)
print(f"Sample saved to {sample_path}")
print("Final Song Output:")
print(sample_song_np)

wandb.finish()
