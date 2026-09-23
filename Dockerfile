FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN mkdir -p /data
ENV OAUTH_DB_PATH=/data/oauth.sqlite3

EXPOSE 8000
CMD ["python", "-m", "screenscribe.app"]
