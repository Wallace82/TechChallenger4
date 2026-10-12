# Perfil demo-cpu: sklearn + FakeChatModel. Sem CUDA, sem Llama, sem HF_TOKEN.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PERFIL_EXECUCAO=demo-cpu \
    ML_RISCO_HABILITADO=1 \
    HOSPITAL_DB_PATH=/app/artifacts/demo/hospital_ui.db \
    ARTIFACTS_DIR=/app/artifacts \
    PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY lib ./lib
COPY scripts ./scripts
COPY referencias ./referencias
COPY tests ./tests
COPY pyproject.toml .env.example ./
COPY artifacts ./artifacts

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/').read()" || exit 1

CMD ["python", "scripts/app.py"]
