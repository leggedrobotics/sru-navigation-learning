#  SPDX-License-Identifier: BSD-3-Clause

import copy

import pytest
import torch

from rsl_rl.algorithms import MDPO, PPO, SPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticSRU


@pytest.mark.parametrize(
    ("algorithm_class", "policy_kind"),
    [
        (PPO, "sru"),
        (SPO, "sru"),
        (MDPO, "sru"),
        (PPO, "gru"),
        (PPO, "lstm"),
        (PPO, "lstm_sru"),
        (PPO, "mlp"),
    ],
)
@torch.no_grad()
def test_bootstrap_preserves_critic_history_between_rollouts(algorithm_class, policy_kind):
    torch.manual_seed(7)
    num_envs, num_actor_obs, num_critic_obs = 4, 19, 36
    policy_kwargs = dict(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
    )
    if policy_kind == "mlp":
        policy = ActorCritic(**policy_kwargs)
    elif policy_kind == "sru":
        policy = ActorCriticSRU(
            **policy_kwargs,
            image_input_dims=(4, 2, 2),
            height_input_dims=(4, 2, 2),
            rnn_hidden_size=8,
            dropout=0.2,
        )
    else:
        policy = ActorCriticRecurrent(**policy_kwargs, rnn_type=policy_kind, rnn_hidden_size=8, dropout=0.2)
    if algorithm_class is MDPO:
        second_policy = copy.deepcopy(policy)
        algorithm = MDPO(policy, second_policy, use_muon=False)
    else:
        algorithm = algorithm_class(policy)
    algorithm.init_storage(num_envs, 1, [num_actor_obs], [num_critic_obs], [2])
    if algorithm_class is MDPO:
        partitions = (
            (policy, algorithm.storage_1, algorithm.indices_1),
            (second_policy, algorithm.storage_2, algorithm.indices_2),
        )
    else:
        partitions = ((policy, algorithm.storage, slice(None)),)

    rewards = torch.zeros(num_envs)
    dones = torch.zeros(num_envs, dtype=torch.bool)
    algorithm.act(torch.randn(num_envs, num_actor_obs), torch.randn(num_envs, num_critic_obs))
    algorithm.process_env_step(rewards, dones, {})

    last_critic_obs = torch.randn(num_envs, num_critic_obs)
    # Reference critics consume the boundary observation only once.
    next_values = [copy.deepcopy(critic).evaluate(last_critic_obs[ids]) for critic, _, ids in partitions]
    algorithm.compute_returns(last_critic_obs)
    for (_, storage, _), expected in zip(partitions, next_values):
        torch.testing.assert_close(storage.returns[0], algorithm.gamma * expected)
        storage.clear()

    # The same observation is the first one in the next rollout.
    algorithm.act(torch.randn(num_envs, num_actor_obs), last_critic_obs)
    algorithm.process_env_step(rewards, dones, {})
    for (_, storage, _), expected in zip(partitions, next_values):
        torch.testing.assert_close(storage.values[0], expected)
