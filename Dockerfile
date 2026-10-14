FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app app.py main.py ./
COPY --chown=app:app templates ./templates
COPY --chown=app:app static ./static
RUN mkdir -p /home/app/.cache /data/labels /data/rawimages \
    && chown -R app:app /home/app /data
RUN chmod 755 /home/app

USER app
EXPOSE 8000

CMD ["gunicorn", "--workers", "1", "--threads", "2", "--bind", "0.0.0.0:8000", "app:app"]
