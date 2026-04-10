import sqlite3
import random

def build_tasks():
    print("Connecting to database to sample 50 random papers...")
    try:
        conn = sqlite3.connect('citation_db.sqlite')
        cursor = conn.cursor()
        
        # Pull 50 random legitimate titles so the Agent NEVER loops searching for missing papers!
        cursor.execute("SELECT title FROM s2_papers WHERE title IS NOT NULL AND length(title) > 10 ORDER BY RANDOM() LIMIT 50")
        papers = cursor.fetchall()
        
        if not papers:
            print("ERROR: No papers found in database.")
            return

        tasks_str = "import json\nfrom pydantic import BaseModel\nfrom typing import List\n\n"
        tasks_str += "class Task(BaseModel):\n    id: str\n    difficulty: str\n    claim: str\n    ground_truth_title: str\n\n"
        tasks_str += "TASKS = [\n"
        
        difficulties = ["easy", "medium", "hard"]
        
        for i, paper in enumerate(papers):
            # Escape safely
            title = paper[0].replace('"', "'").replace('\n', ' ') 
            task_id = f"T{i+1:03d}"
            difficulty = difficulties[i % 3]
            
            if difficulty == "easy":
                claim = f"Search our database for the paper titled \\\"{title}\\\". Submit its ID."
            elif difficulty == "medium":
                claim = f"Find the paper \\\"{title}\\\". Verify its metadata and return its ID."
            else:
                claim = f"Search the database for \\\"{title}\\\", check its citations, and submit its ID."
                
            tasks_str += f"""    Task(
        id="{task_id}",
        difficulty="{difficulty}",
        claim="{claim}",
        ground_truth_title="{title}"
    ),\n"""

        tasks_str += "]\n\n"
        
        # Keep Grader Logic Intact
        tasks_str += """class Grader:
    def __init__(self, task: Task):
        self.task = task
        
    def score(self, submitted_paper_id: str, db_conn) -> float:
        cursor = db_conn.cursor()
        cursor.execute("SELECT title FROM s2_papers WHERE corpus_id = ? OR arxiv_id = ?", (submitted_paper_id, submitted_paper_id))
        res = cursor.fetchone()
        
        if not res:
            cursor.execute("SELECT title FROM arxiv_metadata WHERE arxiv_id = ?", (submitted_paper_id,))
            res = cursor.fetchone()
            
        if res and res[0].strip().lower() == self.task.ground_truth_title.strip().lower():
            return 1.0
        return 0.0
"""
        
        # Write to tasks.py directly
        with open('tasks.py', 'w', encoding='utf-8') as f:
            f.write(tasks_str)
            
        print(f"Successfully generated {len(papers)} guaranteed-solvable dynamic tasks into tasks.py!")
        conn.close()
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    build_tasks()
