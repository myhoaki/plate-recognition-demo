import { Fragment, type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Circle, Image as KImage, Layer, Rect, Stage, Text } from 'react-konva';
import type Konva from 'konva';
import RecognitionTest from './RecognitionTest';
import OCRDatasetReview from './OCRDatasetReview';

type Annotation = {
  id?: number; bbox_xyxy_pixels: number[]; ai_detection: boolean; ai_result?: string | null;
  verified_result?: string | null; detection_confidence?: number | null; ocr_confidence?: number | null;
  country?: string | null; country_confidence?: number | null;
};
type ReviewStatus = 'needs_review' | 'reviewed' | 'verified' | 'rejected' | 'no_plate';
type AppSection = 'production' | 'detector' | 'ocr' | 'recognition';
type DatasetSnapshot = { id: string; task: 'detector' | 'ocr'; split_counts: Record<string, number>; manifest_sha256: string; created_at: string };
type DashboardMetrics = { total_images: number; verified_plates: number; images_needing_review: number; ocr_exact_match_accuracy: number | null; character_error_rate: number | null; detection_recall: number | null };
type ImageRecord = { id: string; filename: string; width: number; height: number; detection_complete: boolean; verified: boolean; review_status: ReviewStatus; verified_at?: string | null; file_url: string; annotations?: Annotation[] };
type Job = { job_id?: string; id?: string; kind?: 'single' | 'batch'; status: string; total: number; completed: number; detected: number; failed: number; error?: string | null; existing?: boolean };
type EpochStats = { epoch: number; is_best?: boolean; [key: string]: number | boolean | undefined };
type TrainingJob = { id: string; status: string; model_type: 'detector' | 'ocr'; cancel_requested: boolean; config: { epochs: number; batch_size: number; image_size?: number; device: string }; split_counts: Record<string, number>; current_epoch: number; total_epochs: number; current_batch: number; total_batches: number; current_val: number; total_val: number; current_test: number; total_test: number; metrics: Record<string, number>; epoch_history: EpochStats[]; best_epoch: number | null; best_score: number | null; best_metrics: Record<string, number>; model_ready: boolean; error?: string | null };
type Box = { bbox_xyxy_pixels: number[]; verified_result: string; ai_detection: boolean; ai_result?: string | null; detection_confidence?: number | null; ocr_confidence?: number | null; country?: string | null; country_confidence?: number | null };
const PAGE_SIZE = 500;

const api = async <T,>(url: string, options?: RequestInit): Promise<T> => {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
};
const percent = (v?: number | null) => v == null ? '—' : `${(v * 100).toFixed(1)}%`;

function PlateCrop({ image, box }: { image: HTMLImageElement | null; box: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const [rawX1, rawY1, rawX2, rawY2] = box;
    if (!canvas || !image || box.length < 4) return;
    const x1 = Math.max(0, Math.floor(Math.min(rawX1, rawX2)));
    const y1 = Math.max(0, Math.floor(Math.min(rawY1, rawY2)));
    const x2 = Math.min(image.naturalWidth, Math.ceil(Math.max(rawX1, rawX2)));
    const y2 = Math.min(image.naturalHeight, Math.ceil(Math.max(rawY1, rawY2)));
    if (x2 <= x1 || y2 <= y1) return;
    canvas.width = x2 - x1;
    canvas.height = y2 - y1;
    canvas.getContext('2d')?.drawImage(image, x1, y1, x2 - x1, y2 - y1, 0, 0, canvas.width, canvas.height);
  }, [image, box]);
  return <canvas className="plate-crop" ref={canvasRef} aria-label="Cropped plate image" />;
}

export default function App() {
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [index, setIndex] = useState(0);
  const [imageTotal, setImageTotal] = useState(0);
  const [pageOffset, setPageOffset] = useState(0);
  const [jumpOrder, setJumpOrder] = useState('1');
  const [section, setSection] = useState<AppSection>('production');
  const source = 'datasets_folder' as const;
  const [syncing, setSyncing] = useState(false);
  const [confidenceReview, setConfidenceReview] = useState<'all' | 'detector' | 'ocr' | 'both'>('all');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [statusFilter, setStatusFilter] = useState<'all' | ReviewStatus>('all');
  const [filteringImages, setFilteringImages] = useState(false);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 800, height: 520 });
  const [zoom, setZoom] = useState(1);
  const [busy, setBusy] = useState(false);
  const [dirtyImageId, setDirtyImageId] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [activeJobs, setActiveJobs] = useState<Job[]>([]);
  const [detectorTrainingJob, setDetectorTrainingJob] = useState<TrainingJob | null>(null);
  const [ocrTrainingJob, setOcrTrainingJob] = useState<TrainingJob | null>(null);
  const [snapshots, setSnapshots] = useState<DatasetSnapshot[]>([]);
  const [datasetMetrics, setDatasetMetrics] = useState<DashboardMetrics | null>(null);
  const [selectedSnapshot, setSelectedSnapshot] = useState({ detector: '', ocr: '' });
  const [trainingConfig, setTrainingConfig] = useState({ epochs: 30, batch_size: 8, image_size: 640, device: 'auto' });
  const [drawing, setDrawing] = useState<number[] | null>(null);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);
  const ocrInputRefs = useRef<Array<HTMLInputElement | null>>([]);
  const imageLoadSequence = useRef(0);
  const stageRef = useRef<Konva.Stage>(null);
  const current = images[index];
  const jobActive = job?.status === 'queued' || job?.status === 'running';
  const batchActive = activeJobs.some((item) => item.kind === 'batch');
  const trainingType = section === 'ocr' ? 'ocr' : 'detector';
  const trainingJob = section === 'ocr' ? ocrTrainingJob : detectorTrainingJob;
  const anyTrainingActive = [detectorTrainingJob, ocrTrainingJob].some((item) => item?.status === 'queued' || item?.status === 'running');
  const trainingActive = trainingJob?.status === 'queued' || trainingJob?.status === 'running';

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      api<{ job: TrainingJob | null }>('/api/training-jobs/latest?model_type=detector'),
      api<{ job: TrainingJob | null }>('/api/training-jobs/latest?model_type=ocr'),
    ]).then(([detector, ocr]) => {
      if (cancelled) return;
      setDetectorTrainingJob(detector.job);
      setOcrTrainingJob(ocr.job);
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, []);

  const refreshSnapshots = useCallback(async () => {
    const data = await api<{ snapshots: DatasetSnapshot[] }>('/api/dataset-snapshots');
    setSnapshots(data.snapshots);
    setSelectedSnapshot((old) => ({
      detector: old.detector || data.snapshots.find((item) => item.task === 'detector')?.id || '',
      ocr: old.ocr || data.snapshots.find((item) => item.task === 'ocr')?.id || '',
    }));
  }, []);
  useEffect(() => { void refreshSnapshots().catch(() => undefined); }, [refreshSnapshots]);
  const refreshDatasetMetrics = useCallback(async () => setDatasetMetrics(await api<DashboardMetrics>('/api/metrics')), []);
  useEffect(() => { void refreshDatasetMetrics().catch(() => undefined); }, [refreshDatasetMetrics]);

  useEffect(() => {
    let cancelled = false;
    const timers: number[] = [];
    const poll = (job: TrainingJob, update: (job: TrainingJob) => void) => {
      if (job.status !== 'queued' && job.status !== 'running') return;
      let timer = 0;
      const refresh = async () => {
        try {
          const state = await api<TrainingJob>(`/api/training-jobs/${job.id}`);
          if (cancelled) return;
          update(state);
          if (state.status === 'queued' || state.status === 'running') timer = window.setTimeout(refresh, 2000);
        } catch { if (!cancelled) timer = window.setTimeout(refresh, 4000); }
      };
      timer = window.setTimeout(refresh, 1000);
      timers.push(timer);
    };
    if (detectorTrainingJob) poll(detectorTrainingJob, setDetectorTrainingJob);
    if (ocrTrainingJob) poll(ocrTrainingJob, setOcrTrainingJob);
    return () => { cancelled = true; timers.forEach((timer) => window.clearTimeout(timer)); };
  }, [detectorTrainingJob?.id, detectorTrainingJob?.status, ocrTrainingJob?.id, ocrTrainingJob?.status]);

  useEffect(() => {
    let cancelled = false;
    const updateQueue = async () => {
      try {
        const result = await api<{ jobs: Job[] }>('/api/detection-jobs/active');
        if (!cancelled) setActiveJobs(result.jobs);
      } catch { /* Keep the last known queue state if a poll fails briefly. */ }
    };
    void updateQueue();
    const timer = window.setInterval(updateQueue, 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    const savedJobId = window.localStorage.getItem('platelens.activeDetectionJob');
    if (!savedJobId) return;
    let cancelled = false;
    api<Job>(`/api/jobs/${savedJobId}`).then((restored) => {
      if (cancelled) return;
      setJob(restored);
      if (restored.status !== 'queued' && restored.status !== 'running') {
        window.localStorage.removeItem('platelens.activeDetectionJob');
        setMessage(`Detection ${restored.status}: ${restored.completed}/${restored.total} images, ${restored.detected} with plates, ${restored.failed} failed.`);
      } else {
        setMessage(`Resuming job status: ${restored.completed}/${restored.total} images completed.`);
      }
    }).catch((e) => {
      if (!cancelled) setMessage(`Could not restore detection status: ${(e as Error).message}`);
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (current) setJumpOrder(String(pageOffset + index + 1));
  }, [current?.id, pageOffset, index]);

  const loadImages = useCallback(async (src = source, preferredId?: string, offset = 0) => {
    const requestId = ++imageLoadSequence.current;
    const params = new URLSearchParams({ source: src, limit: String(PAGE_SIZE), offset: String(offset), review: confidenceReview, threshold: String(confidenceThreshold), review_status: statusFilter });
    const data = await api<{ images: ImageRecord[]; total: number }>(`/api/images?${params}`);
    if (requestId !== imageLoadSequence.current) return;
    setImages(data.images);
    setImageTotal(data.total);
    setPageOffset(offset);
    const nextIndex = preferredId ? data.images.findIndex((item) => item.id === preferredId) : -1;
    setIndex(nextIndex >= 0 ? nextIndex : 0);
  }, [source, confidenceReview, confidenceThreshold, statusFilter]);

  useEffect(() => {
    if (!job?.id || (job.status !== 'queued' && job.status !== 'running')) return;
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      try {
        const state = await api<Job>(`/api/jobs/${job.id}`);
        if (cancelled) return;
        setJob(state);
        if (state.status === 'queued' || state.status === 'running') {
          setMessage(`${state.kind === 'batch' ? 'Batch' : 'Image'} detection: ${state.completed.toLocaleString()}/${state.total.toLocaleString()} images completed.`);
          timer = window.setTimeout(poll, 1500);
          return;
        }
        window.localStorage.removeItem('platelens.activeDetectionJob');
        setMessage(`Detection ${state.status}: ${state.completed}/${state.total} images, ${state.detected} with plates, ${state.failed} failed.`);
        if (current) {
          const detail = await api<ImageRecord>(`/api/images/${current.id}`);
          if (!cancelled) {
            // Keep the reviewed image selected if its updated confidence/status makes it
            // fall out of the active review filter after detection.
            setImages((all) => all.map((item) => item.id === detail.id ? { ...item, ...detail } : item));
            setBoxes((detail.annotations || []).map((a) => ({ ...a, verified_result: a.verified_result || '' })));
          }
        }
      } catch (e) {
        if (!cancelled) {
          setMessage(`Could not update detection status: ${(e as Error).message}`);
          timer = window.setTimeout(poll, 3000);
        }
      }
    };
    timer = window.setTimeout(poll, 900);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [job?.id, job?.status, source, pageOffset, current?.id, loadImages]);

  useEffect(() => {
    let active = true;
    setFilteringImages(true);
    loadImages().catch((e) => setMessage(e.message)).finally(() => {
      if (active) setFilteringImages(false);
    });
    return () => { active = false; };
  }, [loadImages]);
  useEffect(() => {
    setSelectedBoxIndex(null);
    ocrInputRefs.current = [];
    if (!current) { setImageEl(null); setBoxes([]); return; }
    setDirtyImageId(null);
    let cancelled = false;
    setMessage('');
    api<ImageRecord>(`/api/images/${current.id}`).then((detail) => {
      if (!cancelled) setBoxes((detail.annotations || []).map((a) => ({ ...a, verified_result: a.verified_result || '' })));
    }).catch((e) => !cancelled && setMessage(e.message));
    const img = new window.Image();
    img.onload = () => { if (!cancelled) setImageEl(img); };
    img.onerror = () => !cancelled && setMessage('Could not load this image.');
    img.src = `${current.file_url}?v=${current.id}`;
    return () => { cancelled = true; };
  }, [current?.id, current?.file_url]);

  useEffect(() => {
    const fit = () => {
      const maxW = Math.max(320, window.innerWidth - 430);
      const maxH = Math.max(300, window.innerHeight - 190);
      const ratio = current ? current.width / current.height : 1.5;
      let width = Math.min(maxW, 1200), height = width / ratio;
      if (height > maxH) { height = maxH; width = height * ratio; }
      setStageSize({ width, height });
    };
    fit(); window.addEventListener('resize', fit); return () => window.removeEventListener('resize', fit);
  }, [current?.width, current?.height]);

  const scale = useMemo(() => current ? stageSize.width / current.width * zoom : zoom, [current, stageSize.width, zoom]);
  const move = (n: number) => {
    if (n < 0 && index === 0 && pageOffset > 0) {
      const offset = Math.max(0, pageOffset - PAGE_SIZE);
      void loadImages(source, undefined, offset).then(() => setIndex(Math.min(PAGE_SIZE - 1, imageTotal - offset - 1)));
    } else if (n > 0 && index === images.length - 1 && pageOffset + images.length < imageTotal) {
      void loadImages(source, undefined, pageOffset + PAGE_SIZE);
    } else {
      setIndex((i) => Math.max(0, Math.min(images.length - 1, i + n)));
    }
  };
  const jumpToOrder = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const order = Number(jumpOrder);
    if (!Number.isInteger(order) || order < 1 || order > imageTotal) {
      setMessage(`Enter an image number from 1 to ${imageTotal.toLocaleString()}.`);
      return;
    }
    const offset = Math.floor((order - 1) / PAGE_SIZE) * PAGE_SIZE;
    setBusy(true);
    try {
      await loadImages(source, undefined, offset);
      setIndex((order - 1) - offset);
      setMessage('');
    } catch (e) {
      setMessage((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const reloadRegisteredImages = async () => {
    setBusy(true); setMessage('Loading imported dataset…');
    try {
      await loadImages(source, current?.id, pageOffset);
      setMessage('Imported dataset reloaded.');
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const syncDatasetsFolder = async () => {
    setSyncing(true); setMessage('Scanning datasets/imports/ folder…');
    try {
      const result = await api<{ synced: number; total: number }>('/api/images/sync-datasets-folder', { method: 'POST' });
      setMessage(`Sync complete: ${result.synced} new/updated image${result.synced === 1 ? '' : 's'}, ${result.total} total.`);
      await loadImages('datasets_folder', undefined, 0);
    } catch (e) { setMessage((e as Error).message); } finally { setSyncing(false); }
  };
  const detect = async (batch = false) => {
    if (!batch && !current) return;
    setBusy(true); setMessage(batch ? 'Queueing all images…' : 'Queueing detection…');
    try {
      const result = await api<Job>(batch ? '/api/detection-jobs/batch' : '/api/detection-jobs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: batch ? undefined : JSON.stringify({ image_id: current!.id }),
      });
      const id = result.job_id!;
      window.localStorage.setItem('platelens.activeDetectionJob', id);
      setJob({ ...result, id, kind: batch ? 'batch' : 'single' });
      setMessage(result.existing
        ? `A batch is already ${result.status}: ${result.completed.toLocaleString()}/${result.total.toLocaleString()} images completed.`
        : `${batch ? 'Batch' : 'Image'} detection queued for ${result.total.toLocaleString()} image${result.total === 1 ? '' : 's'}. It will continue on the server if you close this page.`);
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const startTraining = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedSnapshot[trainingType]) { setMessage(`Create or select a ${trainingType} dataset snapshot first.`); return; }
    setBusy(true); setMessage(`Queueing ${trainingType} training…`);
    try {
      const job = await api<TrainingJob>('/api/training-jobs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...trainingConfig, model_type: trainingType, dataset_snapshot_id: selectedSnapshot[trainingType] }),
      });
      if (trainingType === 'ocr') setOcrTrainingJob(job);
      else setDetectorTrainingJob(job);
      setMessage(job.status === 'running' ? `${job.model_type} training is already running.` : `${job.model_type} training queued. It continues on the server if you close this page.`);
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const createSnapshot = async (task: 'detector' | 'ocr') => {
    setBusy(true); setMessage(`Creating ${task} dataset snapshot…`);
    try {
      const snapshot = await api<DatasetSnapshot>(`/api/dataset-snapshots?task=${task}`, { method: 'POST' });
      setSnapshots((items) => [snapshot, ...items]);
      setSelectedSnapshot((old) => ({ ...old, [task]: snapshot.id }));
      setMessage(`${task} snapshot created · ${snapshot.split_counts.train} train, ${snapshot.split_counts.val} validation, ${snapshot.split_counts.test} test.`);
    } catch (e) { setMessage((e as Error).message); }
    finally { setBusy(false); }
  };
  const deleteSnapshot = async (snapshot: DatasetSnapshot) => {
    if (!window.confirm(`Delete snapshot ${snapshot.id.slice(0, 8)} (${snapshot.task})? This removes its files and cannot be undone.`)) return;
    setBusy(true); setMessage(`Deleting ${snapshot.task} snapshot…`);
    try {
      await api(`/api/dataset-snapshots/${snapshot.id}`, { method: 'DELETE' });
      setSnapshots(items => items.filter(item => item.id !== snapshot.id));
      setSelectedSnapshot(old => ({ ...old, [snapshot.task]: old[snapshot.task] === snapshot.id ? '' : old[snapshot.task] }));
      setMessage(`${snapshot.task} snapshot deleted.`);
    } catch (e) { setMessage((e as Error).message); }
    finally { setBusy(false); }
  };
  const activateModel = async (target: TrainingJob) => {
    setBusy(true);
    try {
      await api(`/api/training-jobs/${target.id}/activate`, { method: 'POST' });
      setMessage(`Trained ${target.model_type} activated. New detections will use it.`);
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const cancelTraining = async (target: TrainingJob) => {
    if (!trainingActive) return;
    setBusy(true);
    try {
      const stopped = await api<TrainingJob>(`/api/training-jobs/${target.id}/cancel`, { method: 'POST' });
      if (target.model_type === 'ocr') setOcrTrainingJob(stopped);
      else setDetectorTrainingJob(stopped);
      setMessage(stopped.status === 'cancelled' ? 'Training stopped.' : 'Stop requested. Training will stop after the current batch.');
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const save = async () => {
    if (!current) return;
    setBusy(true); setMessage('Saving annotations…');
    try {
      const saved = await api<{ image: ImageRecord }>(`/api/images/${current.id}/annotations`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotations: boxes.map((b) => ({ bbox_xyxy_pixels: b.bbox_xyxy_pixels, verified_result: b.verified_result })) }),
      });
      setMessage('Annotations saved to PostgreSQL.');
      setDirtyImageId(null);
      setImages((all) => all.map((item) => item.id === current.id ? { ...item, verified: false, review_status: saved.image.review_status, verified_at: null } : item));
      setBoxes((saved.image.annotations || []).map((a) => ({ ...a, verified_result: a.verified_result || '' })));
      void refreshDatasetMetrics().catch(() => undefined);
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const setReviewStatus = async (status: ReviewStatus) => {
    if (!current || dirtyImageId === current.id) return;
    setBusy(true);
    try {
      const result = await api<{ image: ImageRecord }>(`/api/images/${current.id}/review-status`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status }),
      });
      setImages((all) => all.map((item) => item.id === current.id ? { ...item, verified: result.image.verified, review_status: result.image.review_status, verified_at: result.image.verified_at } : item));
      setMessage(`Image status set to ${status.replace('_', ' ')}.`);
      void refreshDatasetMetrics().catch(() => undefined);
      await loadImages(source, current.id, pageOffset);
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const markVerified = async () => {
    if (!current) return;
    await setReviewStatus(current.review_status === 'verified' ? 'needs_review' : 'verified');
  };
  const removeBox = (idx: number) => {
    setBoxes((all) => all.filter((_, n) => n !== idx));
    setSelectedBoxIndex((selected) => selected == null || selected === idx ? null : selected > idx ? selected - 1 : selected);
    if (current) setDirtyImageId(current.id);
  };
  const point = () => {
    const p = stageRef.current?.getPointerPosition(); return p ? [p.x / scale, p.y / scale] : null;
  };
  const startDraw = (event: Konva.KonvaEventObject<MouseEvent>) => {
    if (!current || event.target !== stageRef.current) return;
    const p = point(); if (p) setDrawing([p[0], p[1], p[0], p[1]]);
  };
  const updateDraw = () => { if (!drawing || !stageRef.current) return; const p = point(); if (p) setDrawing([drawing[0], drawing[1], p[0], p[1]]); };
  const endDraw = () => {
    if (!drawing) return;
    const box = [Math.min(drawing[0], drawing[2]), Math.min(drawing[1], drawing[3]), Math.max(drawing[0], drawing[2]), Math.max(drawing[1], drawing[3])];
    if (box[2] - box[0] >= 3 && box[3] - box[1] >= 3) {
      setBoxes((old) => [...old, { bbox_xyxy_pixels: box, verified_result: '', ai_detection: false }]);
      setSelectedBoxIndex(boxes.length);
      if (current) setDirtyImageId(current.id);
    }
    setDrawing(null);
  };
  const setText = (idx: number, text: string) => {
    setBoxes((all) => all.map((box, i) => i === idx ? { ...box, verified_result: text } : box));
    if (current) setDirtyImageId(current.id);
  };
  const moveBox = (idx: number, x: number, y: number) => {
    if (!current) return;
    setBoxes((all) => all.map((box, i) => {
      if (i !== idx) return box;
      const [x1, y1, x2, y2] = box.bbox_xyxy_pixels;
      const width = x2 - x1, height = y2 - y1;
      const left = Math.max(0, Math.min(current.width - width, x / scale));
      const top = Math.max(0, Math.min(current.height - height, y / scale));
      return { ...box, bbox_xyxy_pixels: [left, top, left + width, top + height] };
    }));
    setDirtyImageId(current.id);
  };
  const resizeBoxCorner = (idx: number, corner: number, x: number, y: number) => {
    if (!current) return;
    setBoxes((all) => all.map((box, i) => {
      if (i !== idx) return box;
      const coords = [...box.bbox_xyxy_pixels];
      const moveLeft = corner === 0 || corner === 2;
      const moveTop = corner === 0 || corner === 1;
      const xIndex = moveLeft ? 0 : 2;
      const yIndex = moveTop ? 1 : 3;
      coords[xIndex] = Math.max(0, Math.min(current.width, x / scale));
      coords[yIndex] = Math.max(0, Math.min(current.height, y / scale));
      const left = Math.min(coords[0], coords[2]), right = Math.max(coords[0], coords[2]);
      const top = Math.min(coords[1], coords[3]), bottom = Math.max(coords[1], coords[3]);
      if (right - left < 2 || bottom - top < 2) return box;
      return { ...box, bbox_xyxy_pixels: [left, top, right, bottom] };
    }));
    setDirtyImageId(current.id);
  };

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const editing = !!target?.closest('input, textarea, select, [contenteditable="true"]');
      if (section !== 'production') return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
        event.preventDefault(); void save(); return;
      }
      if (editing || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key === 'ArrowRight' || event.key.toLowerCase() === 'n') { event.preventDefault(); move(1); }
      else if (event.key === 'ArrowLeft' || event.key.toLowerCase() === 'p') { event.preventDefault(); move(-1); }
      else if (event.key.toLowerCase() === 'v') { event.preventDefault(); void markVerified(); }
      else if (event.key.toLowerCase() === 't') {
        event.preventDefault();
        const idx = selectedBoxIndex ?? (boxes.length ? 0 : null);
        if (idx != null) { setSelectedBoxIndex(idx); ocrInputRefs.current[idx]?.focus(); }
      } else if (event.key === 'Delete' || event.key === 'Backspace') {
        if (selectedBoxIndex != null) { event.preventDefault(); removeBox(selectedBoxIndex); setSelectedBoxIndex(null); }
      }
    };
    window.addEventListener('keydown', handleShortcut);
    return () => window.removeEventListener('keydown', handleShortcut);
  }, [section, current?.id, boxes.length, selectedBoxIndex, save, markVerified, move, removeBox]);

  const renderTrainingSection = (type: 'detector' | 'ocr', trained: TrainingJob | null) => {
    const isOcr = type === 'ocr';
    const isActive = trained?.status === 'queued' || trained?.status === 'running';
    const otherTrainingActive = [detectorTrainingJob, ocrTrainingJob].some((item) => item?.model_type !== type && (item?.status === 'queued' || item?.status === 'running'));
    const units = isOcr ? 'plate crops' : 'images';
    return <section className={`training-page ${isOcr ? 'ocr-training-page' : 'detector-training-page'}`}>
      <header className="training-page-head">
        <div><h2>{isOcr ? 'OCR training' : 'Detection training'}</h2>
          <p>{isOcr ? 'Train character recognition from cropped license plates.' : 'Train plate localization: find the plate position in each image.'}</p></div>
        {trained?.model_ready && <button onClick={() => void activateModel(trained)} disabled={busy}>Activate {isOcr ? 'OCR' : 'detector'} model</button>}
      </header>
      <form className="training-config" onSubmit={startTraining}>
        <label>Dataset snapshot<select value={selectedSnapshot[type]} onChange={(e) => setSelectedSnapshot((old) => ({ ...old, [type]: e.target.value }))} required>
          <option value="">Create a snapshot first</option>{snapshots.filter((item) => item.task === type).map((item) => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.split_counts.train || 0}/{item.split_counts.val || 0}/{item.split_counts.test || 0} train/val/test</option>)}
        </select></label>
        <button type="button" onClick={() => void createSnapshot(type)} disabled={busy}>Create snapshot</button>
        <label>Epochs<input type="number" min="1" max="300" value={trainingConfig.epochs} onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: Number(e.target.value) })} /></label>
        <label>Batch size<input type="number" min="1" max="64" value={trainingConfig.batch_size} onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: Number(e.target.value) })} /></label>
        {!isOcr && <label>Image size<input type="number" min="320" max="1280" step="32" value={trainingConfig.image_size} onChange={(e) => setTrainingConfig({ ...trainingConfig, image_size: Number(e.target.value) })} /></label>}
        <button className="primary" type="submit" disabled={busy || isActive || otherTrainingActive || !selectedSnapshot[type]}>Start {isOcr ? 'OCR' : 'detector'} training</button>
      </form>
      {otherTrainingActive && <p className="training-source-note">Another training run is active. Wait for it to finish before starting this one.</p>}
      <p className="training-source-note">{isOcr
        ? 'Snapshot source: verified OCR text from imported OCR data and verified plate annotations. Crops from one source image stay in the same split.'
        : 'Snapshot source: verified plate boxes in the selected immutable detector dataset snapshot.'}</p>
      {isOcr && <OCRDatasetReview />}
      {trained ? <section className="training-status" aria-live="polite">
        <div className="training-status-head"><strong>{isOcr ? 'OCR' : 'Detector'} · {trained.status}</strong>
          {isActive && <button className="stop-training" onClick={() => void cancelTraining(trained)} disabled={busy || trained.cancel_requested}>{trained.cancel_requested ? 'Stopping…' : 'Stop training'}</button>}
        </div>
        {trained.split_counts && Object.keys(trained.split_counts).length > 0 && <div className="training-splits">
          <label>Train · epoch {trained.current_epoch} / {trained.total_epochs} · batch {trained.total_batches ? `${trained.current_batch} / ${trained.total_batches}` : 'preparing'} · {Math.min(trained.split_counts.train || 0, trained.current_batch * trained.config.batch_size).toLocaleString()} / {(trained.split_counts.train || 0).toLocaleString()} {units} this epoch
            <progress value={Math.min(trained.split_counts.train || 0, trained.current_batch * trained.config.batch_size)} max={trained.split_counts.train || 1} />
          </label>
          <label>Validation · {Math.min(trained.current_val || 0, trained.total_val || trained.split_counts.val || 0).toLocaleString()} / {(trained.total_val || trained.split_counts.val || 0).toLocaleString()} {units}
            <progress value={trained.current_val || 0} max={trained.total_val || trained.split_counts.val || 1} />
          </label>
          <label>Test · {Math.min(trained.current_test || 0, trained.total_test || trained.split_counts.test || 0).toLocaleString()} / {(trained.total_test || trained.split_counts.test || 0).toLocaleString()} {units}
            <progress value={trained.current_test || 0} max={trained.total_test || trained.split_counts.test || 1} />
          </label>
        </div>}
        {trained.cancel_requested && isActive && <span>Stop requested. Waiting for the current batch to finish…</span>}
        {!isOcr && trained.metrics['metrics/mAP50(B)'] != null && <div className="training-metric-grid">
          <div><small>{isActive ? 'Latest validation' : 'Held-out test'} mAP50</small><b>{percent(trained.metrics['metrics/mAP50(B)'])}</b></div>
          <div><small>mAP50–95</small><b>{percent(trained.metrics['metrics/mAP50-95(B)'])}</b></div>
          <div><small>Precision</small><b>{percent(trained.metrics['metrics/precision(B)'])}</b></div>
          <div><small>Recall</small><b>{percent(trained.metrics['metrics/recall(B)'])}</b></div>
        </div>}
        {isOcr && trained.metrics.val_plate_accuracy != null && <div className="training-metric-grid">
          <div><small>Validation exact plate</small><b>{percent(trained.metrics.val_plate_accuracy)}</b></div>
          <div><small>Validation CER</small><b>{percent(trained.metrics.val_character_error_rate)}</b></div>
          {trained.metrics.test_plate_accuracy != null && <div><small>Test exact plate</small><b>{percent(trained.metrics.test_plate_accuracy)}</b></div>}
          {trained.metrics.test_character_error_rate != null && <div><small>Test CER</small><b>{percent(trained.metrics.test_character_error_rate)}</b></div>}
        </div>}
        {trained.epoch_history?.length > 0 && <div className="epoch-history">
          <strong>Epoch statistics · best epoch {trained.best_epoch ?? '—'}</strong>
          <div className="epoch-history-list">{trained.epoch_history.map((row) => <div className="epoch-history-row" key={row.epoch}>
            <b>Epoch {row.epoch}{row.is_best ? ' · Best' : ''}</b>
            {isOcr
              ? <span>Loss {(Number(row.train_loss) || 0).toFixed(4)} · Exact {percent(Number(row.val_plate_accuracy) || 0)} · CER {percent(Number(row.val_character_error_rate) || 0)}</span>
              : <span>mAP50 {percent(Number(row['metrics/mAP50(B)']) || 0)} · mAP50–95 {percent(Number(row['metrics/mAP50-95(B)']) || 0)} · Precision {percent(Number(row['metrics/precision(B)']) || 0)} · Recall {percent(Number(row['metrics/recall(B)']) || 0)}</span>}
          </div>)}</div>
        </div>}
        {trained.model_ready && <small>Best checkpoint saved as <code>best.pt</code>; future runs start from the retained best weights.</small>}
        {trained.error && <span className="training-error">{trained.error}</span>}
      </section> : <section className="training-status training-empty"><strong>No {isOcr ? 'OCR' : 'detector'} training run yet</strong><span>Configure the run above to begin. Epoch statistics and split progress will appear here.</span></section>}
    </section>;
  };

  return <main className="app-shell">
    <header className="topbar app-header"><div><h1>PlateLens</h1><p>License plate production and training</p></div>
      <nav className="section-tabs" aria-label="Application sections">
        {([['production', 'Verify Datasets'], ['detector', 'Train Detection'], ['ocr', 'Train OCR'], ['recognition', 'Recognition / Test']] as const).map(([id, label]) => <button key={id} className={section === id ? 'active' : ''} onClick={() => { setSection(id); setMessage(''); }}>{label}</button>)}
      </nav>
    </header>
    {section === 'production' && <>
    <div className="production-heading"><div><h2>Verify Datasets</h2><p>Review images and verify plate boxes before creating training snapshots.</p></div>
      <div className="snapshot-actions">
        <button onClick={() => void syncDatasetsFolder()} disabled={busy || syncing}>{syncing ? 'Scanning…' : 'Sync datasets/imports/'}</button>
        <button onClick={() => void createSnapshot('detector')} disabled={busy}>Build detector snapshot</button>
        <button onClick={() => void createSnapshot('ocr')} disabled={busy}>Build OCR snapshot</button>
        <button onClick={() => void detect(true)} disabled={busy || batchActive || anyTrainingActive}>Detect all images</button>
      </div>
    </div>
    <section className="snapshot-list"><strong>Immutable snapshots</strong>{snapshots.length ? snapshots.map((item) => <span key={item.id}><b>{item.task}</b> · {item.id.slice(0, 8)} · {item.split_counts.train}/{item.split_counts.val}/{item.split_counts.test} train/val/test · {item.manifest_sha256.slice(0, 12)} <button className="delete-snapshot" onClick={() => void deleteSnapshot(item)} title="Delete snapshot">×</button></span>) : <span>No snapshots created yet.</span>}</section>
    {datasetMetrics && <section className="metrics-dashboard dataset-dashboard"><div className="metric"><span>Total images</span><strong>{datasetMetrics.total_images.toLocaleString()}</strong></div><div className="metric"><span>Verified plates</span><strong>{datasetMetrics.verified_plates.toLocaleString()}</strong></div><div className="metric"><span>Needs review</span><strong>{datasetMetrics.images_needing_review.toLocaleString()}</strong></div><div className="metric"><span>OCR exact match</span><strong>{percent(datasetMetrics.ocr_exact_match_accuracy)}</strong></div><div className="metric"><span>Character error rate</span><strong>{percent(datasetMetrics.character_error_rate)}</strong></div><div className="metric"><span>Detection recall</span><strong>{percent(datasetMetrics.detection_recall)}</strong></div><button className="metrics-refresh" onClick={() => void refreshDatasetMetrics()} disabled={busy}>Refresh statistics</button></section>}
    {activeJobs.length > 0 && <section className="queue-status" aria-live="polite">
      <strong>Detection queue</strong>{activeJobs.map((item) => <div className="queue-item" key={item.id}>
        <span>{item.kind === 'batch' ? 'Batch' : 'Image'} · {item.status === 'queued' ? 'Waiting' : 'Running'}</span><span>{item.completed.toLocaleString()} / {item.total.toLocaleString()} complete</span>
        <progress value={item.completed} max={item.total || 1} />
      </div>)}
    </section>}
    <section className="toolbar">
      <button onClick={reloadRegisteredImages} disabled={busy}>Reload images</button>
      <label className="review-filter">Review queue
        <select value={confidenceReview} onChange={(e) => { setFilteringImages(true); setConfidenceReview(e.target.value as typeof confidenceReview); }}>
          <option value="all">All · low confidence first</option><option value="detector">Detector confidence</option><option value="ocr">OCR confidence</option><option value="both">Detector or OCR</option>
        </select>
      </label>
      <label className="review-filter">Status filter
        <select value={statusFilter} onChange={(e) => { setFilteringImages(true); setStatusFilter(e.target.value as typeof statusFilter); }}>
          <option value="all">All statuses</option><option value="needs_review">Needs review</option><option value="reviewed">Reviewed</option><option value="verified">Verified</option><option value="rejected">Rejected</option><option value="no_plate">No plate</option>
        </select>
      </label>
      {confidenceReview !== 'all' && <label className="review-filter">Below
        <select value={confidenceThreshold} onChange={(e) => { setFilteringImages(true); setConfidenceThreshold(Number(e.target.value)); }}>
          <option value={0.5}>50%</option><option value={0.6}>60%</option><option value={0.7}>70%</option><option value={0.8}>80%</option><option value={0.9}>90%</option>
        </select>
      </label>}
      {filteringImages && <span className="filtering-indicator" role="status"><span className="filter-spinner" />Filtering images…</span>}
      <div className="image-review-controls" aria-label="Image review navigation">
        <button aria-label="Previous image" title="Previous image" onClick={() => move(-1)} disabled={!images.length || (!pageOffset && index === 0)}>←</button>
        <span className="counter">{images.length ? `${pageOffset + index + 1} / ${imageTotal}` : 'No images'}</span>
        <button aria-label="Next image" title="Next image" onClick={() => move(1)} disabled={!images.length || pageOffset + index >= imageTotal - 1}>→</button>
        {current && <select className={`review-status-action status-${current.review_status}`} aria-label="Set current image review status" title="Set review status" value={current.review_status} onChange={(e) => void setReviewStatus(e.target.value as ReviewStatus)} disabled={busy || dirtyImageId === current.id}>
          <option value="needs_review">Needs review</option><option value="reviewed">Reviewed</option><option value="verified">Verified</option><option value="rejected">Rejected</option><option value="no_plate">No plate</option>
        </select>}
      </div>
      <form className="jump-form" onSubmit={jumpToOrder}>
        <label htmlFor="jump-order">Go to image</label>
        <input id="jump-order" type="number" min="1" max={imageTotal || 1} value={jumpOrder} onChange={(e) => setJumpOrder(e.target.value)} disabled={!imageTotal || busy} />
        <button type="submit" disabled={!imageTotal || busy}>Go</button>
      </form>
      <span className="filename">{current?.filename || (source === 'datasets_folder' ? 'No images found — click "Sync from datasets/ folder" to scan.' : 'Import images into datasets/raw/')}</span>
      <div className="zoom-controls" aria-label="Image zoom controls">
        <button aria-label="Zoom out" title="Zoom out" onClick={() => setZoom((value) => Math.max(0.25, Math.round((value - 0.25) * 100) / 100))} disabled={zoom <= 0.25}>−</button>
        <span>{Math.round(zoom * 100)}%</span>
        <button aria-label="Zoom in" title="Zoom in" onClick={() => setZoom((value) => Math.min(4, Math.round((value + 0.25) * 100) / 100))} disabled={zoom >= 4}>+</button>
        <button onClick={() => setZoom(1)} disabled={zoom === 1}>Fit</button>
      </div>
      {current && dirtyImageId === current.id && <span className="verify-badge pending">Unsaved edits</span>}
      <button className="primary" onClick={() => detect(false)} disabled={busy || jobActive || anyTrainingActive || !current}>Auto detect</button>
      <button className="primary" onClick={save} disabled={busy || !current}>Save labels</button>
    </section>
    <section className="workspace">
      <div className="canvas-wrap">
        {current && imageEl ? <Stage ref={stageRef} width={stageSize.width * zoom} height={stageSize.height * zoom} onMouseDown={startDraw} onMouseMove={updateDraw} onMouseUp={endDraw} onTouchStart={startDraw as never} onTouchMove={updateDraw} onTouchEnd={endDraw}>
          <Layer><KImage image={imageEl} width={current.width * scale} height={current.height * scale} listening={false} />
            {boxes.map((box, i) => { const [x1,y1,x2,y2] = box.bbox_xyxy_pixels; const handles = [[x1,y1],[x2,y1],[x1,y2],[x2,y2]]; return <Fragment key={i}>
              <Rect x={x1*scale} y={y1*scale} width={(x2-x1)*scale} height={(y2-y1)*scale} stroke={box.ai_detection ? '#f59e0b' : '#16a34a'} strokeWidth={2} draggable onClick={() => setSelectedBoxIndex(i)} onTap={() => setSelectedBoxIndex(i)} onDragStart={() => setSelectedBoxIndex(i)} onDragEnd={(e) => moveBox(i, e.target.x(), e.target.y())} onContextMenu={(e) => { e.evt.preventDefault(); removeBox(i); }} />
              <Text x={x1*scale} y={Math.max(0,y1*scale-21)} text={box.ai_result || box.verified_result || `Plate ${i+1}`} fontSize={14} fill="white" padding={3} listening={false} />
              {selectedBoxIndex === i && handles.map(([hx, hy], corner) => <Circle key={corner} x={hx*scale} y={hy*scale} radius={6} fill="#38bdf8" stroke="#e0f2fe" strokeWidth={1} draggable onMouseDown={(e) => e.cancelBubble = true} onTouchStart={(e) => e.cancelBubble = true} onDragEnd={(e) => resizeBoxCorner(i, corner, e.target.x(), e.target.y())} />)}
            </Fragment>; })}
            {drawing && <Rect x={Math.min(drawing[0],drawing[2])*scale} y={Math.min(drawing[1],drawing[3])*scale} width={Math.abs(drawing[2]-drawing[0])*scale} height={Math.abs(drawing[3]-drawing[1])*scale} stroke="#38bdf8" dash={[5,3]} />}
          </Layer>
        </Stage> : <div className="empty">{current ? 'Loading image…' : (source === 'datasets_folder' ? 'No images loaded. Click "Sync from datasets/ folder" above.' : 'No registered images found. Import a dataset into datasets/raw/ first.')}</div>}
      </div>
      <aside className="side-panel"><h2>Plate annotations <span>{boxes.length}</span></h2><p className="hint">Draw a box on the image. Right-click a box row to remove it.</p>
        <div className="annotation-list">{boxes.map((box, i) => <article className="annotation" key={i} onContextMenu={(e) => { e.preventDefault(); removeBox(i); }}>
          <div className="annotation-head"><strong>Plate {i + 1}</strong><span className={box.ai_detection ? 'ai-tag' : 'manual-tag'}>{box.ai_detection ? 'AI' : 'Manual'}</span></div>
          <PlateCrop image={imageEl} box={box.bbox_xyxy_pixels} />
          <label>Verified plate text<input ref={(element) => { ocrInputRefs.current[i] = element; }} onFocus={() => setSelectedBoxIndex(i)} value={box.verified_result} onChange={(e) => setText(i, e.target.value.toUpperCase())} placeholder={box.ai_result || 'Enter text'} /></label>
          {box.ai_detection && <div className="stats"><span>AI guess <b>{box.ai_result || '—'}</b></span><span>OCR confidence <b>{percent(box.ocr_confidence)}</b></span><span>Detection confidence <b>{percent(box.detection_confidence)}</b></span><span>Country <b>{box.country || 'Unknown'} · {percent(box.country_confidence)}</b></span></div>}
        </article>)}</div>
        {job && (job.status === 'running' || job.status === 'queued') && <div className="job"><div>{job.kind === 'batch' ? 'Batch detection' : 'Image detection'} {job.completed}/{job.total}</div><progress value={job.completed} max={job.total || 1} /></div>}
      </aside>
    </section>
    {message && <div className="status" role="status">{message}</div>}
    <footer>AI boxes are orange; manual boxes are green. OCR guesses are never saved as ground truth automatically. Save label edits before changing review status. Shortcuts: ←/→ or P/N navigate · Ctrl/⌘+S save · V verify · Delete remove selected box · T focus plate text.</footer>
    </>}
    {section === 'detector' && renderTrainingSection('detector', detectorTrainingJob)}
    {section === 'ocr' && renderTrainingSection('ocr', ocrTrainingJob)}
    {section === 'recognition' && <RecognitionTest />}
    {message && section !== 'production' && <div className="status" role="status">{message}</div>}
  </main>;
}
