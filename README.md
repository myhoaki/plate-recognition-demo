# PlateLens

A self-hosted, still-image license-plate annotation workspace. AI detection, OCR, and optional detector training run as background jobs; users can review or draw plate boxes, verify the OCR text, and save annotations to PostgreSQL.

## Architecture

| Component | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Annotation canvas | Konva.js / React Konva |
| Backend API | Python FastAPI |
| Database | PostgreSQL |
| Image storage | Docker bind-mounted filesystem |
| Background jobs | Redis + Celery |

## Run on a Linux server

Install Docker Engine and the Docker Compose plugin, copy this project to the server, then run from this directory:

```bash
PUID="$(id -u)" PGID="$(id -g)" docker compose up -d --build
```

Open `http://SERVER_IP:8000`. The first detection downloads the pretrained plate detector and OCR model, so the server needs internet access for that initial run. To watch the services:

```bash
docker compose ps
docker compose logs -f api worker
```

To apply code changes, rebuild and restart:

```bash
PUID="$(id -u)" PGID="$(id -g)" docker compose up -d --build
```

## Images and saved data

- Put source images in `rawimages/`. The API reads them without modifying them.
- Uploads are stored in `docker/storage/uploads/`.
- Model cache files are stored in `docker/models/`.
- Detector training datasets and checkpoints are stored in `docker/training/` and are not included in Git.
- PostgreSQL data and Redis queue data are held in named Docker volumes.
- Existing JSON annotations in `docker/labels/` are imported into PostgreSQL at API startup. They are kept as a backup and are not deleted.

The application prevents duplicate raw-image entries by the pair of source and filename. Replacing a file with the same filename updates its registered image and clears stale annotations for that changed image.

## Environment

Optional settings can be placed in a `.env` file beside `compose.yml`:

```env
WEB_PORT=8000
POSTGRES_PASSWORD=choose-a-long-random-password
PUID=1000
PGID=1000
```

Set `PUID` and `PGID` to the output of `id -u` and `id -g` on the server account that owns `docker/storage/` and `docker/models/`. Keep `.env` private. The default database password is intended only for local development; configure a private password before exposing the service to a network.

## Workflow

1. Select an image from `rawimages/` or upload a still image.
2. Use **Auto detect** for one image or **Detect all raw images** for the folder.
3. Review AI boxes and OCR results. Orange boxes are AI detections; green boxes are manually drawn.
4. Draw any missing plate boxes on the image. Right-click a box to remove it.
5. Enter or correct each verified plate text, then save. A blank value accepts the matching AI guess.
6. Click **Mark verified** after reviewing the saved annotations. The image shows a verified badge; editing or rerunning detection clears that mark.

Each annotation stores the box coordinates, AI OCR guess, user-verified text, detector confidence, OCR confidence, country/region estimate, and country/region confidence when available. Batch detection status is shown in the workspace.

## Train a detector

Mark reviewed images as **Verified**, then select **Train detector**. Training uses verified images and their saved boxes, with a deterministic 80/10/10 train/validation/test split. The small YOLO11n pretrained detector is fine-tuned; epoch and batch size are configurable, and validation metrics are recorded each epoch. After training, the held-out test split is evaluated. Use **Use this model** to make the trained detector active; OCR remains the existing FastALPR model. Training artifacts and the active checkpoint are kept in `docker/training/`.

The training worker downloads the pretrained YOLO weights on its first run and needs internet access. This Compose setup installs CPU-only PyTorch; training can take a long time on large datasets. Detection jobs wait while training is active.

## Dataset importer

`import_dataset.py` can copy supported still images into `rawimages/` from the extracted Kaggle car-plate dataset. For example:

```bash
python3 import_dataset.py ~/Downloads/car-plate-detection
```

Restart or refresh the app after importing so it syncs the new files.

## Development

The React app can be run independently with `cd frontend && npm install && npm run dev`. The FastAPI service is in `backend/app/`; Docker Compose starts it with PostgreSQL, Redis, and the Celery worker. `app.py`, `templates/`, and `static/` are retained as the earlier Flask implementation and are not used by the current Compose stack.
