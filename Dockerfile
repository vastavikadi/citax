FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Preload the dataset-backed SQLite database so the Space can answer quickly on first request.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ruby56/Citation-Database', filename='citation_db.sqlite', repo_type='dataset', local_dir='/app')"

COPY . .

EXPOSE 7860

# Run Hugging Face Space entrypoint
CMD ["python", "hf_entry.py"]