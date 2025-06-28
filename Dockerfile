FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    HF_HOME=/home/user/.cache/huggingface \
    XDG_CACHE_HOME=/home/user/.cache \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user src/ ./src/

RUN mkdir -p $TRANSFORMERS_CACHE $HOME/.streamlit && \
    chown -R user:user $HOME

USER user

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]