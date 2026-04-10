import json
import os
from typing import Dict, List

from openai import OpenAI


class HybridCitationEvaluator:
    def __init__(self):
        self.api_key = os.environ.get("HF_TOKEN")
        self.base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        self.model_name = os.environ.get("MODEL_NAME", "google/gemma-4-9b-it")

    @staticmethod
    def _programmatic_score(final_reward: float, trajectory: List[Dict[str, str]], step_count: int, max_steps: int) -> Dict[str, float]:
        parsed_actions = []
        for step in trajectory:
            try:
                parsed_actions.append(json.loads(step["action"]))
            except Exception:
                parsed_actions.append({})

        action_types = [a.get("action_type", "") for a in parsed_actions]

        valid = 0
        for action in parsed_actions:
            action_type = action.get("action_type")
            if action_type in {"search", "read_abstract", "get_citations", "submit"}:
                valid += 1
        validity = valid / len(parsed_actions) if parsed_actions else 0.0

        exact_match = 1.0 if final_reward >= 1.0 else 0.0
        used_search = 1.0 if "search" in action_types else 0.0
        used_evidence = 1.0 if ("read_abstract" in action_types or "get_citations" in action_types) else 0.0
        used_submit = 1.0 if "submit" in action_types else 0.0
        workflow = (0.5 * used_search) + (0.25 * used_evidence) + (0.25 * used_submit)

        if max_steps <= 1:
            efficiency = 1.0
        else:
            efficiency = max(0.0, 1.0 - ((step_count - 1) / (max_steps - 1)))

        programmatic = (
            0.55 * exact_match
            + 0.20 * workflow
            + 0.15 * validity
            + 0.10 * efficiency
        )

        return {
            "exact_match": round(exact_match, 4),
            "workflow": round(workflow, 4),
            "validity": round(validity, 4),
            "efficiency": round(efficiency, 4),
            "score": round(programmatic, 4),
        }

    def _llm_score(self, task_claim: str, trajectory: List[Dict[str, str]], final_reward: float) -> Dict[str, str]:
        if not self.api_key:
            return {
                "score": 0.5,
                "feedback": "LLM scoring skipped because HF_TOKEN is not set.",
            }

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        prompt = {
            "task_claim": task_claim,
            "final_reward": final_reward,
            "trajectory": trajectory,
            "rubric": {
                "evidence_usage": "Did the agent inspect meaningful search/metadata evidence?",
                "consistency": "Do actions logically lead to the final submission?",
                "risk_awareness": "Did the agent avoid clearly invalid jumps?",
            },
            "output_format": {
                "score": "float between 0 and 1",
                "feedback": "one concise paragraph",
            },
        }

        system = (
            "You are grading an RL citation agent. Return only JSON with keys score and feedback. "
            "Score must be a float in [0, 1]."
        )

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```json"):
                raw = raw[7:-3].strip()
            elif raw.startswith("```"):
                raw = raw[3:-3].strip()
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            feedback = str(parsed.get("feedback", "No feedback provided."))
            return {"score": score, "feedback": feedback}
        except Exception as e:
            return {"score": 0.5, "feedback": f"LLM scoring fallback used: {e}"}

    def evaluate(
        self,
        task_claim: str,
        final_reward: float,
        trajectory: List[Dict[str, str]],
        step_count: int,
        max_steps: int,
        include_llm_score: bool = True,
    ) -> Dict[str, object]:
        programmatic = self._programmatic_score(final_reward, trajectory, step_count, max_steps)
        llm = self._llm_score(task_claim, trajectory, final_reward) if include_llm_score else {
            "score": 0.0,
            "feedback": "LLM scoring disabled by configuration.",
        }

        llm_weight = 0.30 if include_llm_score else 0.0
        program_weight = 1.0 - llm_weight
        overall = (program_weight * float(programmatic["score"])) + (llm_weight * float(llm["score"]))

        return {
            "overall": round(overall, 4),
            "programmatic": programmatic,
            "llm": {
                "score": round(float(llm["score"]), 4),
                "feedback": llm["feedback"],
            },
        }
