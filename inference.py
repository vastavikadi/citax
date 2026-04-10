import argparse
import os
from dotenv import load_dotenv

from evaluation import HybridCitationEvaluator
from rl_agent import QLearningCitationAgent
from tasks import TASKS

load_dotenv()


def _stratified_task_ids(limit: int) -> list[str]:
    by_diff = {"easy": [], "medium": [], "hard": []}
    for task in TASKS:
        by_diff.setdefault(task.difficulty, []).append(task.id)

    selected: list[str] = []
    idx = 0
    ordering = ["easy", "medium", "hard"]
    while len(selected) < limit:
        progressed = False
        for diff in ordering:
            bucket = by_diff.get(diff, [])
            if idx < len(bucket):
                selected.append(bucket[idx])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        idx += 1
    return selected


def run_benchmark(
    episodes_per_task: int = 120,
    include_llm_score: bool = False,
    checkpoint_path: str = "rl_policy.json",
    load_checkpoint: bool = False,
    save_checkpoint: bool = True,
    skip_train: bool = False,
    force_retrain: bool = False,
    train_task_limit: int | None = None,
    eval_task_limit: int | None = None,
    ci_fast: bool = False,
) -> dict:
    if ci_fast:
        episodes_per_task = min(episodes_per_task, 6)
        include_llm_score = False
        if train_task_limit is None:
            train_task_limit = 12
        if eval_task_limit is None:
            eval_task_limit = 6

    all_task_ids = [task.id for task in TASKS]
    train_ids = _stratified_task_ids(train_task_limit) if train_task_limit else all_task_ids
    eval_ids = _stratified_task_ids(eval_task_limit) if eval_task_limit else all_task_ids

    if load_checkpoint and os.path.exists(checkpoint_path) and not force_retrain:
        agent = QLearningCitationAgent.load(checkpoint_path)
        checkpoint_msg = f"[CHECKPOINT] loaded={checkpoint_path}"
    else:
        agent = QLearningCitationAgent()
        checkpoint_msg = "[CHECKPOINT] initialized=new-policy"

    print(checkpoint_msg, flush=True)

    train_stats = {"episodes": 0, "average_reward": 0.0, "solved_rate": 0.0}
    if not skip_train:
        stats = agent.train_on_tasks(task_ids=train_ids, episodes_per_task=episodes_per_task)
        train_stats = {
            "episodes": stats.episodes,
            "average_reward": round(stats.average_reward, 4),
            "solved_rate": round(stats.solved_rate, 4),
        }

    if save_checkpoint:
        agent.save(checkpoint_path)
        print(f"[CHECKPOINT] saved={checkpoint_path}", flush=True)

    evaluator = HybridCitationEvaluator()
    results = {}

    print(f"[TRAIN] episodes={train_stats['episodes']} avg_reward={train_stats['average_reward']:.4f} solved_rate={train_stats['solved_rate']:.4f}", flush=True)

    task_map = {task.id: task for task in TASKS}
    for task_id in eval_ids:
        task = task_map[task_id]
        final_reward, trajectory, steps, max_steps = agent.rollout(task.id)
        evaluation = evaluator.evaluate(
            task_claim=task.claim,
            final_reward=final_reward,
            trajectory=trajectory,
            step_count=steps,
            max_steps=max_steps,
            include_llm_score=include_llm_score,
        )
        results[task.id] = evaluation
        print(
            f"[EVAL] task={task.id} reward={final_reward:.2f} overall={evaluation['overall']:.4f}"
            f" programmatic={evaluation['programmatic']['score']:.4f} llm={evaluation['llm']['score']:.4f}",
            flush=True,
        )

    avg = sum(v["overall"] for v in results.values()) / len(results) if results else 0.0
    solved = sum(1 for v in results.values() if v["programmatic"]["exact_match"] >= 1.0)
    print(f"[SUMMARY] average_overall={avg:.4f} solved={solved}/{len(results)}", flush=True)

    return {
        "train": train_stats,
        "config": {
            "episodes_per_task": episodes_per_task,
            "include_llm_score": include_llm_score,
            "ci_fast": ci_fast,
            "checkpoint_path": checkpoint_path,
            "train_task_count": len(train_ids),
            "eval_task_count": len(eval_ids),
        },
        "evaluation": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the Citation RL agent.")
    parser.add_argument("--episodes-per-task", type=int, default=120)
    parser.add_argument("--include-llm-score", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default="rl_policy.json")
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--train-task-limit", type=int, default=None)
    parser.add_argument("--eval-task-limit", type=int, default=None)
    parser.add_argument("--ci-fast", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        episodes_per_task=args.episodes_per_task,
        include_llm_score=args.include_llm_score,
        checkpoint_path=args.checkpoint_path,
        load_checkpoint=args.load_checkpoint,
        save_checkpoint=not args.no_save_checkpoint,
        skip_train=args.skip_train,
        force_retrain=args.force_retrain,
        train_task_limit=args.train_task_limit,
        eval_task_limit=args.eval_task_limit,
        ci_fast=args.ci_fast,
    )
