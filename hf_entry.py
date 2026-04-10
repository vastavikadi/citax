import json
import os
from typing import Tuple

import gradio as gr

from evaluation import HybridCitationEvaluator
from rl_agent import QLearningCitationAgent
from tasks import TASKS


CHECKPOINT_PATH = "rl_policy.json"
_cached_agent: QLearningCitationAgent | None = None


def _load_or_train_agent(
    episodes_per_task: int,
    checkpoint_path: str,
    load_checkpoint: bool,
    force_retrain: bool,
    ci_fast: bool,
) -> Tuple[QLearningCitationAgent, str]:
    global _cached_agent
    if ci_fast:
        episodes_per_task = min(episodes_per_task, 30)

    if _cached_agent is not None:
        return _cached_agent, "Loaded in-memory policy."

    if load_checkpoint and os.path.exists(checkpoint_path) and not force_retrain:
        _cached_agent = QLearningCitationAgent.load(checkpoint_path)
        return _cached_agent, f"Loaded checkpoint from {checkpoint_path}."

    agent = QLearningCitationAgent()
    task_ids = [task.id for task in TASKS]
    if ci_fast:
        task_ids = [task.id for task in TASKS[:12]]
    stats = agent.train_on_tasks(task_ids=task_ids, episodes_per_task=episodes_per_task)
    agent.save(checkpoint_path)
    _cached_agent = agent
    return (
        _cached_agent,
        f"Trained new policy. episodes={stats.episodes}, avg_reward={stats.average_reward:.4f}, solved_rate={stats.solved_rate:.4f}, checkpoint={checkpoint_path}",
    )


def run_task(
    task_id: str,
    episodes_per_task: int = 80,
    include_llm_score: bool = False,
    checkpoint_path: str = CHECKPOINT_PATH,
    load_checkpoint: bool = True,
    force_retrain: bool = False,
    ci_fast: bool = False,
) -> str:
    agent, train_msg = _load_or_train_agent(
        episodes_per_task=episodes_per_task,
        checkpoint_path=checkpoint_path,
        load_checkpoint=load_checkpoint,
        force_retrain=force_retrain,
        ci_fast=ci_fast,
    )
    task = next((t for t in TASKS if t.id == task_id), TASKS[0])

    final_reward, trajectory, step_count, max_steps = agent.rollout(task.id)
    evaluator = HybridCitationEvaluator()
    eval_result = evaluator.evaluate(
        task_claim=task.claim,
        final_reward=final_reward,
        trajectory=trajectory,
        step_count=step_count,
        max_steps=max_steps,
        include_llm_score=(include_llm_score and not ci_fast),
    )

    payload = {
        "train": train_msg,
        "config": {
            "checkpoint_path": checkpoint_path,
            "load_checkpoint": load_checkpoint,
            "force_retrain": force_retrain,
            "ci_fast": ci_fast,
        },
        "task": {"id": task.id, "difficulty": task.difficulty, "claim": task.claim},
        "episode": {
            "final_reward": final_reward,
            "step_count": step_count,
            "max_steps": max_steps,
            "trajectory": trajectory,
        },
        "evaluation": eval_result,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_agent(user_input: str) -> str:
    candidate = user_input.strip().upper()
    if candidate in {task.id for task in TASKS}:
        return run_task(task_id=candidate)
    return run_task(task_id=TASKS[0].id)


def _build_ui() -> gr.Blocks:
    with gr.Blocks(title="Citation RL Agent") as demo:
        gr.Markdown("# Citation RL Agent\nTrain/evaluate a reinforcement-learning citation policy with hybrid scoring.")

        with gr.Row():
            task_dropdown = gr.Dropdown(
                choices=[t.id for t in TASKS],
                value=TASKS[0].id,
                label="Task ID",
            )
            episodes_slider = gr.Slider(
                minimum=20,
                maximum=200,
                step=10,
                value=80,
                label="Training Episodes Per Task",
            )
            llm_checkbox = gr.Checkbox(value=False, label="Enable LLM scoring")
            ci_fast_checkbox = gr.Checkbox(value=False, label="CI Fast Mode")

        with gr.Row():
            checkpoint_path = gr.Textbox(value=CHECKPOINT_PATH, label="Checkpoint Path")
            load_checkpoint = gr.Checkbox(value=True, label="Load Checkpoint")
            force_retrain = gr.Checkbox(value=False, label="Force Retrain")

        run_button = gr.Button("Run RL Evaluation")
        output = gr.Code(label="Result JSON", language="json")

        run_button.click(
            fn=run_task,
            inputs=[task_dropdown, episodes_slider, llm_checkbox, checkpoint_path, load_checkpoint, force_retrain, ci_fast_checkbox],
            outputs=[output],
        )

    return demo


if __name__ == "__main__":
    app = _build_ui()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))