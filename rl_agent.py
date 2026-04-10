import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from environment import Action, CitationEnv, Observation


@dataclass
class TrainStats:
    episodes: int
    average_reward: float
    solved_rate: float


class QLearningCitationAgent:
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 0.95,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        max_candidate_papers: int = 3,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.max_candidate_papers = max_candidate_papers
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(dict)

    @staticmethod
    def _action_signature(action: Action) -> str:
        payload = action.model_dump(exclude_none=True)
        parts = [f"{k}={payload[k]}" for k in sorted(payload.keys())]
        return "|".join(parts)

    @staticmethod
    def _extract_claim_title(claim: str) -> str:
        match = re.findall(r'\"([^\"]+)\"', claim)
        if match:
            return match[0]
        match = re.findall(r'“([^”]+)”', claim)
        if match:
            return match[0]
        return claim.strip()

    @staticmethod
    def _difficulty_bucket(claim: str) -> str:
        lower = claim.lower()
        if "check its citations" in lower:
            return "hard"
        if "verify its metadata" in lower:
            return "medium"
        return "easy"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in re.split(r"[^a-zA-Z0-9]+", text.lower()) if tok]

    def _overlap_score(self, claim_title: str, paper_title: str) -> float:
        lhs = set(self._tokenize(claim_title))
        rhs = set(self._tokenize(paper_title))
        if not lhs or not rhs:
            return 0.0
        return len(lhs.intersection(rhs)) / len(lhs)

    def state_key(self, obs: Observation) -> str:
        difficulty = self._difficulty_bucket(obs.current_claim)
        has_results = int(bool(obs.search_results))
        has_abstract = int(bool(obs.last_abstract))
        has_citations = int(bool(obs.citations_data))
        step_bucket = min(obs.step_count, 6)
        return (
            f"d={difficulty}|step={step_bucket}|sr={has_results}|"
            f"ab={has_abstract}|ct={has_citations}"
        )

    def candidate_actions(self, obs: Observation) -> List[Action]:
        difficulty = self._difficulty_bucket(obs.current_claim)
        title = self._extract_claim_title(obs.current_claim)
        tokens = [t for t in re.split(r"\s+", title) if t]
        shorter_query = " ".join(tokens[:6]) if tokens else title
        broader_query = " ".join(tokens[:3]) if tokens else title

        candidates: List[Action] = [
            Action(action_type="search", query=title),
            Action(action_type="search", query=shorter_query),
            Action(action_type="search", query=broader_query),
        ]

        results = obs.search_results or []
        if results:
            ranked = sorted(
                results,
                key=lambda row: self._overlap_score(title, row.get("title", "")),
                reverse=True,
            )
            for row in ranked[: self.max_candidate_papers]:
                arxiv_id = (row.get("arxiv_id") or "").strip()
                corpus_id = (row.get("corpus_id") or "").strip()
                if arxiv_id and arxiv_id.lower() not in {"none", "nan"}:
                    candidates.append(Action(action_type="read_abstract", paper_id=arxiv_id))
                    candidates.append(Action(action_type="submit", paper_id=arxiv_id))
                if corpus_id and corpus_id.lower() not in {"none", "nan"}:
                    if difficulty == "hard":
                        candidates.append(Action(action_type="get_citations", paper_id=corpus_id))
                    candidates.append(Action(action_type="submit", paper_id=corpus_id))

        # A safe fallback candidate so the policy can terminate if it gets stuck.
        if obs.step_count >= 3:
            candidates.append(Action(action_type="submit", paper_id="unknown"))

        dedup: Dict[str, Action] = {}
        for action in candidates:
            dedup[self._action_signature(action)] = action
        return list(dedup.values())

    def _initial_q(self, obs: Observation, action: Action) -> float:
        action_type = action.action_type
        if action_type == "search":
            return 0.08
        if action_type in {"read_abstract", "get_citations"}:
            return 0.06 if obs.search_results else -0.02
        if action_type == "submit":
            has_evidence = bool(obs.last_abstract or obs.citations_data)
            return 0.02 if has_evidence else -0.08
        return -0.10

    def _pick_action(self, obs: Observation, greedy: bool = False) -> Action:
        state = self.state_key(obs)
        candidates = self.candidate_actions(obs)

        for action in candidates:
            sig = self._action_signature(action)
            if sig not in self.q_table[state]:
                self.q_table[state][sig] = self._initial_q(obs, action)

        should_explore = (not greedy) and (random.random() < self.epsilon)
        if should_explore:
            return random.choice(candidates)

        best_score = max(self.q_table[state].values())
        best_sigs = [sig for sig, score in self.q_table[state].items() if score == best_score]
        best_sig = random.choice(best_sigs)
        for action in candidates:
            if self._action_signature(action) == best_sig:
                return action
        return random.choice(candidates)

    def _update(
        self,
        state: str,
        action_sig: str,
        reward: float,
        next_state: str,
        next_actions: List[Action],
        done: bool,
    ) -> None:
        current_q = self.q_table[state].get(action_sig, 0.0)
        if done:
            target = reward
        else:
            next_values = []
            for action in next_actions:
                next_sig = self._action_signature(action)
                next_values.append(self.q_table[next_state].get(next_sig, 0.0))
            target = reward + self.gamma * (max(next_values) if next_values else 0.0)

        self.q_table[state][action_sig] = current_q + self.alpha * (target - current_q)

    def train_on_tasks(self, task_ids: List[str], episodes_per_task: int = 100) -> TrainStats:
        rewards: List[float] = []
        solved = 0

        for task_id in task_ids:
            for _ in range(episodes_per_task):
                env = CitationEnv(task_id=task_id)
                obs = env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    state = self.state_key(obs)
                    action = self._pick_action(obs)
                    action_sig = self._action_signature(action)

                    next_obs, reward, done, _ = env.step(action)
                    episode_reward += reward.value

                    next_state = self.state_key(next_obs)
                    next_actions = self.candidate_actions(next_obs)
                    self._update(state, action_sig, reward.value, next_state, next_actions, done)
                    obs = next_obs

                rewards.append(episode_reward)
                if episode_reward > 0:
                    solved += 1

                if self.epsilon > self.epsilon_min:
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        episodes = len(rewards)
        return TrainStats(
            episodes=episodes,
            average_reward=sum(rewards) / episodes if episodes else 0.0,
            solved_rate=(solved / episodes) if episodes else 0.0,
        )

    def rollout(self, task_id: str) -> Tuple[float, List[Dict[str, str]], int, int]:
        env = CitationEnv(task_id=task_id)
        obs = env.reset()
        done = False
        final_reward = 0.0
        trajectory: List[Dict[str, str]] = []

        while not done:
            action = self._pick_action(obs, greedy=True)
            next_obs, reward, done, _ = env.step(action)
            final_reward = reward.value if done else final_reward
            trajectory.append(
                {
                    "observation": obs.model_dump_json(),
                    "action": action.model_dump_json(exclude_none=True),
                    "reward": f"{reward.value:.4f}",
                    "done": str(done),
                }
            )
            obs = next_obs

        return final_reward, trajectory, obs.step_count, env.max_steps

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_decay": self.epsilon_decay,
                    "epsilon_min": self.epsilon_min,
                    "max_candidate_papers": self.max_candidate_papers,
                    "q_table": self.q_table,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "QLearningCitationAgent":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        agent = cls(
            alpha=data.get("alpha", 0.25),
            gamma=data.get("gamma", 0.95),
            epsilon=data.get("epsilon", 0.10),
            epsilon_decay=data.get("epsilon_decay", 0.995),
            epsilon_min=data.get("epsilon_min", 0.05),
            max_candidate_papers=data.get("max_candidate_papers", 3),
        )
        raw_q = data.get("q_table", {})
        agent.q_table = defaultdict(dict, {k: dict(v) for k, v in raw_q.items()})
        return agent
