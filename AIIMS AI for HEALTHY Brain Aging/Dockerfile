FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEUROSAARTHI_ALLOW_NETWORK=false

RUN useradd --create-home --uid 10001 neurosaarthi
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY neurosaarthi ./neurosaarthi
COPY data_contracts ./data_contracts
COPY demo ./demo
COPY etl ./etl
COPY evaluation ./evaluation
COPY features ./features
COPY harmonization ./harmonization
COPY models ./models
RUN python -m pip install --upgrade pip && python -m pip install ".[imaging]"

USER neurosaarthi
ENTRYPOINT ["neurosaarthi"]
CMD ["--version"]
