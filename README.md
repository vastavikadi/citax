---
title: Citation Benchmark
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Citation-Agent: RL Citation Benchmark

Citation-Agent is an OpenEnv-compliant environment for training and evaluating a reinforcement learning citation agent on academic literature review and citation selection.

Given a claim, the agent must navigate a localized research database to search across papers, read abstracts, traverse citation graphs, and ultimately select the correct paper that supports the claim.

---

## The Challenge

Most agent benchmarks focus on web scraping or bash terminal navigation. This environment tests an agent's ability to reason over complex scholarly graphs. The agent must:
- Differentiate between ArXiv IDs and Semantic Scholar Corpus IDs.
- Follow citation intents.
- Run SQL-backed queries to find core papers.
- Avoid getting stuck in infinite search loops.

---

## RL Architecture

- `environment.py`: RL environment with actions, transitions, and dense reward logic.
- `tasks.py`: 50 tasks with deterministic grader checks.
- `rl_agent.py`: Q-learning policy for citation actions.
- `evaluation.py`: Hybrid evaluation with programmatic checks and optional LLM scoring.
- `hf_entry.py`: Gradio app entrypoint for Hugging Face Spaces deployment.

---

## How to Run

### Option 1: Running Locally (Recommended)

1. **Install Dependencies:**
   Requires Python 3.10+.
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables:**
   We use the Hugging Face Router with the `google/gemma-4-31B-it` model.
    - PowerShell:
      ```powershell
      $env:HF_TOKEN="your_huggingface_api_key"
      $env:MODEL_NAME="google/gemma-4-31B-it"
      $env:API_BASE_URL="https://router.huggingface.co/v1"
      ```
    - Bash/Zsh:
      ```bash
      export HF_TOKEN="your_huggingface_api_key"
      export MODEL_NAME="google/gemma-4-31B-it"
      export API_BASE_URL="https://router.huggingface.co/v1"
      ```

3. **Train + Evaluate the RL Agent (CLI):**
   ```bash
   python inference.py --episodes-per-task 120
   ```
   Add `--include-llm-score` to enable LLM-based trajectory scoring.

    Useful controls:
    - Resume from saved checkpoint:
       ```bash
       python inference.py --load-checkpoint --checkpoint-path rl_policy.json
       ```
    - Evaluate only (skip train):
       ```bash
       python inference.py --load-checkpoint --skip-train
       ```
    - Force a fresh retrain:
       ```bash
       python inference.py --force-retrain
       ```
    - CI fast mode (reduced train/eval set, no LLM scoring):
       ```bash
       python inference.py --ci-fast
       ```

4. **Run the Hugging Face app locally:**
   ```bash
   python hf_entry.py
   ```
   Then open `http://localhost:7860`.

### Option 2: Running via Docker (Hugging Face Spaces validation)

To test if your environment works with the automated Hugging Face Spaces pipeline:

```bash
docker build -t citation-agent .
docker run --env HF_TOKEN="your_token" --env MODEL_NAME="google/gemma-4-31B-it" --env API_BASE_URL="https://router.huggingface.co/v1" citation-agent
```

### Option 3: Deploy Directly to Hugging Face Space

1. Create a Space ID (for example `your-username/citation-rl-agent`).
2. Set your token:
   - PowerShell:
     ```powershell
     $env:HF_TOKEN="your_hf_token"
     ```
3. Deploy:
   ```bash
   python deploy_hf_space.py --space-id your-username/citation-rl-agent
   ```
4. Optional private Space:
   ```bash
   python deploy_hf_space.py --space-id your-username/citation-rl-agent --private
   ```

After upload, open `https://huggingface.co/spaces/your-username/citation-rl-agent`.

---

## OpenEnv Compliance Checklist

1. **Real-world task**: Accurately simulates literature review and academic source verification.
2. **OpenEnv Spec Compliance**: 
   - Contains a valid `openenv.yaml`.
   - `environment.py` uses Pydantic Models (`Observation`, `Action`, `Reward`) and implements the required `reset()`, `state()`, and `step(action)` methods.
   - **Observation Space**: JSON containing `current_claim`, `search_results`, `last_abstract`, `citations_data`, `message`, and `step_count`.
   - **Action Space**: `action_type` (`search`, `read_abstract`, `get_citations`, `submit`), `query`, and `paper_id`.
3. **50 Distinct Tasks with Graders**: 
   - `tasks.py` loads 50 distinct Easy, Medium, and Hard tasks from the SQLite database.
   - Includes a deterministic `Grader` class that checks the submitted `paper_id` against the ground truth.
4. **Database Architecture**: 
   - To stay under the 1GB limit for free HF Spaces, `environment.py` uses `huggingface-hub` to automatically pull the `citation_db.sqlite` database from the `ruby56/Citation-Database` dataset on startup.
5. **Continuous Reward Function**: 
   - Uses dense shaping. The agent gets a -0.05 step penalty to encourage efficiency, and -0.1 to -0.2 penalties for formatting errors. It gets the final grader score (+1.0 max) on submit.
6. **Hybrid Evaluation**:
   - Uses deterministic programmatic checks and optional LLM scoring to produce the final benchmark score.
7. **Production Dockerfile**: 
   - Standard Dockerfile that installs dependencies and runs `hf_entry.py` directly for Hugging Face Spaces.
