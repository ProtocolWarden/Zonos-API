FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

RUN apt update && \
    apt install -y \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY zonos ./zonos

RUN pip install uv && \
    uv pip install --system -e . && \
    uv pip install --system -e .[compile]

COPY . ./

CMD ["python3", "api.py"]
