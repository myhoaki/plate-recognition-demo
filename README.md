# PlateLens

A self-hosted license-plate dataset review, model training, and recognition workspace. It uses the existing React/Vite frontend, FastAPI API, PostgreSQL metadata store, and Redis/Celery background worker.

## Run

From this directory, build and start the stack:

```bash
PUID="$(id -u)" PGID="$(id -g)" docker compose up -d --build
```

Open `http://SERVER_IP:8000`. First inference downloads the YOLO26 and PaddleOCR model assets, so the server needs network access for first use. Docker Compose uses CPU-only PyTorch; do not start training until the datasets and parameters are ready.

## Four sections

- **Verify Datasets** reviews registered images from `datasets/raw/`, lets you draw and adjust license-plate boxes, correct OCR text, set a review state, and save. Images load in pages of 500. AI OCR output is never stored as verified text automatically.
- **Train Detection** selects a frozen detector snapshot and trains the YOLO detector. The saved snapshot has train, validation, and test splits, YOLO labels, a manifest, and a SHA-256 hash. Training jobs keep their checkpoints and report the metrics produced by the trainer.
- **Train OCR** selects a frozen OCR snapshot. Samples come from verified OCR annotations imported into the registry or verified plate boxes and text saved in the review section. The CRNN trainer reports validation and test exact-string accuracy and CER. Crops from one source image are assigned to one split.
- **Recognition / Test** uploads one vehicle image, lets you select completed detector/OCR checkpoints, runs the full pipeline, and displays boxes, crops, confidence values, and elapsed time. It can also run a queued batch evaluation against a detector snapshot's fixed test split. Matching uses greedy one-to-one IoU ≥ 0.50. Text comparison trims leading/trailing whitespace and ignores case while preserving all other characters. End-to-end accuracy counts a labeled test image as correct only when all plates match and there are no false detections.

Detection uses Ultralytics YOLO26 and character recognition uses PaddleOCR PP-OCRv6, run as separate stages. YOLO26 pretrained on COCO does not detect license plates, so a detector must be trained and activated before recognition returns any boxes; until then the API returns an explanatory error rather than an empty result. A blank model selection uses the active trained checkpoint when present, otherwise the pretrained default. Checkpoint options only include completed training runs.

## Dataset registry and imports

`datasets/` is the canonical managed store. Imported image bytes are copied once to `datasets/raw/<hash-prefix>/<sha256>.<ext>`. PostgreSQL stores stable sample IDs, source provenance, review state, annotations, and current split assignments. JSON manifests under `datasets/manifests/` retain version history. The verification UI lists only registered samples; files copied into unrelated folders or stale legacy database rows are not displayed. Never edit a manifest by hand.

Import an extracted YOLO detector dataset or an OCR image set with `annotations.csv`. Import runs are dry-run by default. Review the JSON report and conflicts, then pass `--apply` to register the data:

```bash
docker compose exec api python -m app.dataset_registry import /datasets/imports/plates-v1 \
  --source-dataset plates-v1 --import-batch plates-v1-2026-01 --task detector
docker compose exec api python -m app.dataset_registry import /datasets/imports/plates-v1 \
  --source-dataset plates-v1 --import-batch plates-v1-2026-01 --task detector --apply
```

For example, stage a dataset and run the dry-run command:

```bash
mkdir -p datasets/imports/plates-v1
cp -a /path/to/extracted-yolo-dataset/. datasets/imports/plates-v1/
docker compose exec api python -m app.dataset_registry import /datasets/imports/plates-v1 \
  --source-dataset plates-v1 --import-batch plates-v1-2026-01 --task detector
```

After reviewing the report, repeat with `--apply`. Staging is temporary; after successful registration and hash verification, you can remove a copied staging folder because the registry keeps its own content-addressed originals. YOLO layouts with `images/` and `labels/` under `train`, `valid`/`val`, and `test` are supported. Four-coordinate boxes and segmentation polygons are converted to plate bounding boxes. Source classes are mapped to the app's single `license_plate` position class, with original class IDs and the mapping recorded in import provenance. A matching `groups.csv` with `image_path,group_id` keeps frames from the same capture or vehicle together. If no group information is supplied, the audit flags the split leakage limitation. Exact duplicate bytes reuse one registry sample. Cross-split exact duplicates stop a default import; `--conflict-policy prefer-holdout` explicitly assigns each repeated image to its most protected split (`test`, then `val`, then `train`) and records that resolution in the import report. Conflicting annotations or group assignments still stop the transaction. `--conflict-policy keep-existing` keeps current registry annotations/splits. Near-duplicate matches are candidates only and are never automatically merged.

OCR imports require `annotations.csv` with `image_path,plate_text`. Optional columns are `verified`, `is_crop`, `object_index`, `x1,y1,x2,y2`, and `group_id`. Set `verified=true` only when the transcription is ground truth. For a full scene, supply a YOLO sidecar plus `object_index`, or pixel coordinates; for an already cropped plate use `is_crop=true`. The CRNN currently supports its configured uppercase/digit alphabet and up to 12 characters; unsupported text is preserved in the registry but cannot be used by that trainer until its alphabet is extended.

Useful registry commands:

```bash
docker compose exec api python -m app.dataset_registry audit --task detector
docker compose exec api python -m app.dataset_registry manifest --task detector
docker compose exec api python -m app.dataset_registry assign SAMPLE_ID --task detector --split train
docker compose exec api python -m app.dataset_registry assign SAMPLE_ID --task detector --split test --confirm-test-change
docker compose exec api python -m app.dataset_registry snapshot --task detector
curl -X POST 'http://localhost:8000/api/dataset-snapshots?task=ocr'
```

Audit reports include split counts and percentages, missing/changed files, malformed boxes, duplicate and near-duplicate candidates, source/batch counts, OCR character/length distributions, and manifest/snapshot hash changes. Reassigning a test sample requires an explicit flag; related samples inherit the group assignment. New verified samples join train after the first manifest. The registry never silently reshuffles prior assignments.

Training snapshots are independent, copied, hash-verified, and made read-only under `datasets/snapshots/detector/<id>/` and `datasets/snapshots/ocr/<id>/`. Detector snapshots include a YOLO `data.yaml`, per-split labels, export checksums, source manifest version, and an audit manifest. OCR snapshots include split CSVs and sample-level provenance/hash metadata. Select the snapshot in the relevant training section; snapshot creation does not start training. Existing snapshots remain unchanged when annotations or source data change.

Back up both `datasets/` and the PostgreSQL Docker volume. The database holds current assignments/provenance; the JSON manifests and snapshot directories hold portable history and training inputs. Keep `datasets/`, source images, and checkpoints out of Git. The legacy `import_dataset.py` remains available for its earlier review-folder workflow; use the registry CLI for reproducible training data.

## Storage and privacy

- Production/review images: registered imports in `datasets/raw/`
- Canonical originals, reports, manifests, and snapshots: `datasets/`
- Imported OCR images and annotation CSV: `raw-dataset-ocr/`
- Verified snapshots: `verified-dataset-detector/` and `verified-dataset-ocr/`
- User uploads: `docker/storage/uploads/`
- Model cache and checkpoints: `docker/models/` and `docker/training/`
- PostgreSQL and Redis: named Docker volumes

Configure `POSTGRES_PASSWORD`, `PUID`, and `PGID` in a private `.env` file before exposing the application beyond a local development network.

## Implementation notes

Detection boxes and OCR text share the existing annotation records. Dataset snapshots freeze source files and split assignments. OCR snapshots preserve crop IDs, source image IDs, split, and verification state in their manifest. Dataset splitting keeps crops from one source image together. Batch evaluation runs in Celery and stores its selected snapshot/checkpoint IDs with the resulting metrics and incorrect examples.

The project retains the older Flask files for compatibility; Docker Compose serves the React/FastAPI stack.
