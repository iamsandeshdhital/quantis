# QUANTIS — QUantum Advanced Nanoscale Technology & Information Systems
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./

RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

COPY src/ ./src/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY examples/ ./examples/
COPY benchmarks/ ./benchmarks/
COPY configs/ ./configs/
COPY LICENSE ./

RUN mkdir -p results

ENTRYPOINT ["python"]
CMD ["-m", "qit.cli", "--demo"]
