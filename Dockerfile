# Use explicit python 3.10 slim image
FROM python:3.10-slim

LABEL maintainer="Hackathon Team"
LABEL description="Customer Support Ticket Routing — OpenEnv environment"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Core environment files
COPY env.py        ./env.py
COPY tasks.py      ./tasks.py
COPY inference.py  ./inference.py
COPY openenv.yaml  ./openenv.yaml
# FIX: added missing web app files
COPY app.py        ./app.py
COPY index.html    ./index.html
COPY style.css     ./style.css
COPY script.js     ./script.js

ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""
ENV TASK_NAME="easy"
ENV PYTHONUNBUFFERED=1

# FIX: expose port for FastAPI dashboard
EXPOSE 7860

# Default: run inference directly.
# For the web dashboard: docker run ... uvicorn app:app --host 0.0.0.0 --port 7860
CMD ["python", "inference.py"]
