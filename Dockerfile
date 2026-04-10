FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Run Hugging Face Space entrypoint
CMD ["python", "hf_entry.py"]