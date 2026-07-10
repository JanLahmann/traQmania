"""Double-DQN training loop over vectorized environments, in pure numpy.

Works with any Q-function implementing the :class:`~traqmania.agents.base.QFunction`
protocol: the target network is just a second flat parameter vector that gets
swapped in to evaluate target Q-values, so quantum and classical backends share
this loop unchanged.

Env protocol: ``env.reset() -> obs (n_envs, F)``;
``env.step(actions (n_envs,) int) -> (obs, reward (n_envs,), done (n_envs,) bool, info)``,
where done sub-envs are auto-reset (the returned obs is the fresh one — safe here
because done transitions never bootstrap from next_obs).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


class Adam:
    """Standard bias-corrected Adam optimizer on a flat parameter vector."""

    def __init__(
        self,
        n_params: int,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """One descent step; returns the updated parameter vector."""
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class _ReplayBuffer:
    """Uniform ring-buffer replay memory backed by preallocated numpy arrays."""

    def __init__(self, capacity: int, n_features: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_features))
        self.action = np.zeros(capacity, dtype=np.intp)
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, n_features))
        self.done = np.zeros(capacity)
        self._ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        n = obs.shape[0]
        idx = (self._ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_obs[idx] = next_obs
        self.done[idx] = done
        self._ptr = (self._ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)


class DQNTrainer:
    """Double-DQN with epsilon-greedy exploration, replay, and a numpy Adam optimizer."""

    def __init__(
        self,
        qfunc: Any,
        env: Any,
        training_cfg: dict,
        rng: np.random.Generator | None = None,
        env_factory: Callable[[], Any] | None = None,
        stop_event: Any = None,
    ) -> None:
        """``env_factory`` (optional): zero-arg callable returning a FRESH eval env;
        when given, a greedy eval runs every ``eval_every`` (default 50) episodes and
        ``train()`` leaves ``qfunc`` at the best-scoring snapshot.  ``stop_event``
        (optional): ``threading.Event``-like; when set, ``train()`` returns early.
        """
        self.qfunc = qfunc
        self.env = env
        self.cfg = training_cfg
        self.rng = rng if rng is not None else np.random.default_rng(training_cfg.get("seed"))
        self.env_factory = env_factory
        self.stop_event = stop_event
        self.eval_every = int(training_cfg.get("eval_every", 50))
        self.eval_episodes = int(training_cfg.get("eval_episodes", 4))

        self.gamma = training_cfg["gamma"]
        self.batch_size = training_cfg["batch_size"]
        self.target_sync_every = training_cfg["target_sync_every"]
        self.epsilon_start = training_cfg["epsilon_start"]
        self.epsilon_end = training_cfg["epsilon_end"]
        self.epsilon_decay_episodes = training_cfg["epsilon_decay_episodes"]

        self.buffer = _ReplayBuffer(training_cfg["replay_size"], qfunc.n_features)
        self.optimizer = Adam(qfunc.get_params().shape[0], lr=training_cfg["lr"])
        self.target_params = qfunc.get_params()
        self._updates = 0

    def _epsilon(self, episode: int) -> float:
        frac = min(1.0, episode / max(1, self.epsilon_decay_episodes))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def _select_actions(self, obs: np.ndarray, epsilon: float) -> np.ndarray:
        greedy = np.argmax(self.qfunc.q_values(obs), axis=1)
        random_a = self.rng.integers(self.qfunc.n_actions, size=obs.shape[0])
        explore = self.rng.random(obs.shape[0]) < epsilon
        return np.where(explore, random_a, greedy)

    def _update(self) -> float:
        """One double-DQN gradient step on a uniform replay batch; returns MSE loss."""
        idx = self.rng.integers(0, self.buffer.size, size=self.batch_size)
        obs = self.buffer.obs[idx]
        action = self.buffer.action[idx]
        reward = self.buffer.reward[idx]
        next_obs = self.buffer.next_obs[idx]
        done = self.buffer.done[idx]
        rows = np.arange(self.batch_size)

        # Double DQN: online net picks a*, target net evaluates it.
        a_star = np.argmax(self.qfunc.q_values(next_obs), axis=1)
        online_params = self.qfunc.get_params()
        self.qfunc.set_params(self.target_params)
        q_next = self.qfunc.q_values(next_obs)[rows, a_star]
        self.qfunc.set_params(online_params)
        target = reward + self.gamma * (1.0 - done) * q_next

        q_sel = self.qfunc.q_values(obs)[rows, action]
        td = q_sel - target
        upstream = 2.0 * td / self.batch_size  # d(MSE)/d(Q_sel)
        grad = self.qfunc.grad_selected(obs, action, upstream)
        self.qfunc.set_params(self.optimizer.step(online_params, grad))

        self._updates += 1
        if self._updates % self.target_sync_every == 0:
            self.target_params = self.qfunc.get_params()
        return float(np.mean(td**2))

    def _greedy_eval(self, env, max_steps: int = 5000) -> tuple[int, float, float]:
        """Run ``eval_episodes`` greedy (epsilon = 0) episodes on ``env``.

        Returns (laps_completed, best_lap_seconds, mean_return); best_lap is
        +inf when no lap finished.  Expects the racing-env info dict (``lap``,
        ``last_lap_time``); envs without it simply score (0, inf, mean_return).
        """
        obs = env.reset()
        finished = 0
        laps = 0
        best_lap = float("inf")
        return_acc = np.zeros(obs.shape[0])
        episode_returns: list[float] = []
        for _ in range(max_steps):
            actions = np.argmax(self.qfunc.q_values(obs), axis=1)
            obs, reward, done, info = env.step(actions)
            return_acc += reward
            if isinstance(info, dict) and "lap" in info:
                lap_times = np.asarray(info.get("last_lap_time", np.nan), dtype=np.float64)
                if np.any(~np.isnan(lap_times)):
                    best_lap = min(best_lap, float(np.nanmin(lap_times)))
                laps += int(np.sum(np.asarray(info["lap"])[np.flatnonzero(done)]))
            for i in np.flatnonzero(done):
                episode_returns.append(float(return_acc[i]))
            return_acc[done] = 0.0
            finished += int(np.sum(done))
            if finished >= self.eval_episodes:
                break
        mean_return = float(np.mean(episode_returns)) if episode_returns else float("-inf")
        return laps, best_lap, mean_return

    def _eval_snapshot(self, best: dict | None, episode: int) -> dict:
        """Greedy-eval the current params on a fresh env; keep the best snapshot.

        Score is lexicographic (laps_completed, -best_lap, mean_return): more
        finished laps wins, ties broken by the faster lap, then by eval return —
        the return tie-breaker matters before the first lap, where (0, inf)
        would otherwise tie forever and pin the earliest (untrained) snapshot.
        """
        laps, best_lap, mean_return = self._greedy_eval(self.env_factory())
        score = (laps, -best_lap, mean_return)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "params": self.qfunc.get_params(),
                "episode": episode,
                "laps": laps,
                "best_lap": None if np.isinf(best_lap) else best_lap,
            }
        return best

    def train(
        self,
        episodes: int | None = None,
        callback: Callable[[int, dict], None] | None = None,
    ) -> dict:
        """Run until ``episodes`` sub-env episodes have completed; returns history.

        With ``env_factory`` set, a greedy eval runs every ``eval_every`` episodes
        (plus once at the end); ``qfunc`` is left at the BEST-scoring params and
        ``history["best_eval"]`` reports {episode, laps, best_lap}.
        """
        episodes = episodes if episodes is not None else self.cfg["episodes"]
        t_start = time.perf_counter()

        obs = self.env.reset()
        n_envs = obs.shape[0]
        return_acc = np.zeros(n_envs)
        episode_returns: list[float] = []
        losses: list[float] = []
        best: dict | None = None
        evals_done = 0

        while len(episode_returns) < episodes:
            if self.stop_event is not None and self.stop_event.is_set():
                break
            epsilon = self._epsilon(len(episode_returns))
            actions = self._select_actions(obs, epsilon)
            next_obs, reward, done, _info = self.env.step(actions)

            self.buffer.add(obs, actions, reward, next_obs, done)
            return_acc += reward
            obs = next_obs

            # ONE gradient update per env decision-step once the buffer is warm.
            if self.buffer.size >= self.batch_size:
                losses.append(self._update())

            for i in np.flatnonzero(done):
                episode_returns.append(float(return_acc[i]))
                if callback is not None:
                    last_loss = losses[-1] if losses else float("nan")
                    callback(
                        len(episode_returns) - 1,
                        dict(returns=episode_returns[-1], epsilon=epsilon, loss=last_loss),
                    )
            return_acc[done] = 0.0

            if self.env_factory is not None:
                while len(episode_returns) >= (evals_done + 1) * self.eval_every:
                    evals_done += 1
                    best = self._eval_snapshot(best, len(episode_returns))

        history = {
            "episode_returns": episode_returns,
            "losses": losses,
        }
        if self.env_factory is not None:
            best = self._eval_snapshot(best, len(episode_returns))  # final params compete too
            self.qfunc.set_params(best["params"])
            history["best_eval"] = {k: best[k] for k in ("episode", "laps", "best_lap")}
        history["wall_time_s"] = time.perf_counter() - t_start
        return history
