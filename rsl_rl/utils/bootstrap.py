#  SPDX-License-Identifier: BSD-3-Clause

import torch


@torch.no_grad()
def evaluate_bootstrap_value(actor_critic, observations: torch.Tensor) -> torch.Tensor:
    """Evaluate the last observation while preserving the critic's rollout state."""
    if not actor_critic.is_recurrent:
        return actor_critic.evaluate(observations).detach()

    memory = actor_critic.memory_c
    # Both recurrent memory implementations replace the state during evaluation.
    saved_state = memory.hidden_states
    try:
        return actor_critic.evaluate(observations).detach()
    finally:
        memory.hidden_states = saved_state
