"""Backend-agnostic training loops (shared between classical and quantum agents)."""

from traqmania.agents.training.dqn import Adam, DQNTrainer

__all__ = ["Adam", "DQNTrainer"]
