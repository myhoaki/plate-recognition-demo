import { type FormEvent, useCallback, useEffect, useRef, useState } from 'react';
import RecognitionTest from './RecognitionTest';
import OCRDatasetReview from './OCRDatasetReview';
import OCRTextReview from './OCRTextReview';
import AllSamples from './AllSamples';
import AnnotationCanvas, { type AnnotationCanvasHandle } from './AnnotationCanvas';
import PlateCrop from './PlateCrop';

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
type TrainingJob = { id: string; status: string; model_type: 'detector' | 'ocr'; cancel_requested: boolean; config: { epochs: number; batch_size: number; image_size?: number; workers?: number; device: string }; split_counts: Record<string, number>; current_epoch: number; total_epochs: number; current_batch: number; total_batches: number; current_val: number; total_val: number; current_test: number; total_test: number; metrics: Record<string, number>; epoch_history: EpochStats[]; best_epoch: number | null; best_score: number | null; best_metrics: Record<string, number>; model_ready: boolean; error?: string | null };
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

export default function App() {
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [index, setIndex] = useState(0);
  const [imageTotal, setImageTotal] = useState(0);
  const [pageOffset, setPageOffset] = useState(0);
  const [jumpOrder, setJumpOrder] = useState('1');
  const [section, setSection] = useState<AppSection>('production');
const [verifyView, setVerifyView] = useState<'detection' | 'ocr' | 'samples'>('detection');
  const source = 'datasets_folder' as const;
  const [syncing, setSyncing] = useState(false);
  const [confidenceReview, setConfidenceReview] = useState<'all' | 'detector' | 'ocr' | 'both'>('all');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [statusFilter, setStatusFilter] = useState<'all' | ReviewStatus>('all');
  const [filteringImages, setFilteringImages] = useState(false);
  const [boxes, setBoxes] = useState<Box[]>([]);
const [undoStack, setUndoStack] = useState<Box[][]>([]);
const [redoStack, setRedoStack] = useState<Box[][]>([]);
const [mode, setMode] = useState<'select' | 'draw'>('select');
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);
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
  const [trainingConfig, setTrainingConfig] = useState({ epochs: 30, batch_size: 8, image_size: 640, workers: 2, device: 'auto' });
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);
  const ocrInputRefs = useRef<Array<HTMLInputElement | null>>([]);
  const imageLoadSequence = useRef(0);
  const canvasApi = useRef<AnnotationCanvasHandle>(null);
  const editSnapshot = useRef<Box[] | null>(null);
  const [zoomLabel, setZoomLabel] = useState(100);
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

  const openImageInDetection = async (imageId: string) => {
    try {
      await loadImages(source, imageId, pageOffset);
      setSelectedBoxIndex(null);
    } catch (e) { setMessage((e as Error).message); }
  };

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

  const move = (n: number) => {
    if (hasUnsaved && !window.confirm('This image has unsaved annotation edits. Discard them and move on?')) return;
    if (n < 0 && index === 0 && pageOffset > 0) {
      const offset = Math.max(0, pageOffset - PAGE_SIZE);
      void loadImages(source, undefined, offset).then(() => setIndex(Math.min(PAGE_SIZE - 1, imageTotal - offset - 1)));
    } else if (n > 0 && index === images.length - 1 && pageOffset + images.length < imageTotal) {
      void loadImages(source, undefined, pageOffset + PAGE_SIZE);
    } else {
      setIndex((i) => Math.max(0, Math.min(images.length - 1, i + n)));
    }
    setUndoStack([]); setRedoStack([]); setSelectedBoxIndex(null);
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
  const save = async (andNext = false) => {
    if (!current) return;
    setBusy(true); setMessage('Saving annotations…');
    try {
      const saved = await api<{ image: ImageRecord }>(`/api/images/${current.id}/annotations`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotations: boxes.map((b) => ({ bbox_xyxy_pixels: b.bbox_xyxy_pixels, verified_result: b.verified_result })) }),
      });
      setDirtyImageId(null);
      setUndoStack([]); setRedoStack([]);
      setImages((all) => all.map((item) => item.id === current.id ? { ...item, verified: false, review_status: saved.image.review_status, verified_at: null } : item));
      setBoxes((saved.image.annotations || []).map((a) => ({ ...a, verified_result: a.verified_result || '' })));
      void refreshDatasetMetrics().catch(() => undefined);
      // Only report success once the backend has confirmed it.
      setMessage(`Saved to PostgreSQL. Status is now "${saved.image.review_status.replace('_', ' ')}".`);
      if (andNext) move(1);
    } catch (e) {
      setMessage(`Save failed: ${(e as Error).message} Your edits are still unsaved.`);
    } finally { setBusy(false); }
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

  /** Apply a completed annotation edit and record it as one undo step. */
  // A drag pushes several live onChange updates; onEditStart captures the state
  // from before the drag so undo returns there rather than to a mid-drag frame.
  const commitBoxes = (next: Box[]) => {
    const base = editSnapshot.current ?? boxes;
    editSnapshot.current = null;
    setUndoStack((stack) => [...stack.slice(-49), base]);
    setRedoStack([]);
    setBoxes(next);
    if (current) setDirtyImageId(current.id);
  };
  const beginEdit = () => { editSnapshot.current = boxes; };
  const undo = () => {
    if (!undoStack.length) return;
    const previous = undoStack[undoStack.length - 1];
    setRedoStack((stack) => [boxes, ...stack].slice(0, 50));
    setUndoStack((stack) => stack.slice(0, -1));
    setBoxes(previous);
    if (current) setDirtyImageId(current.id);
    setMessage('Undid the last annotation change.');
  };
  const redo = () => {
    if (!redoStack.length) return;
    const next = redoStack[0];
    setUndoStack((stack) => [...stack, boxes].slice(-50));
    setRedoStack((stack) => stack.slice(1));
    setBoxes(next);
    if (current) setDirtyImageId(current.id);
    setMessage('Redid the annotation change.');
  };
  /** Guard navigation so unsaved box edits are never silently dropped. */
  const hasUnsaved = !!current && dirtyImageId === current.id;
  const removeBox = (idx: number) => {
    commitBoxes(boxes.filter((_, n) => n !== idx));
    setSelectedBoxIndex((selected) => selected == null || selected === idx ? null : selected > idx ? selected - 1 : selected);
  };
  const setText = (idx: number, text: string) => {
    setBoxes((all) => all.map((box, i) => i === idx ? { ...box, verified_result: text } : box));
    if (current) setDirtyImageId(current.id);
  };

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const editing = !!target?.closest('input, textarea, select, [contenteditable="true"]');
      if (section !== 'production' || verifyView !== 'detection') return;
      const modifier = event.ctrlKey || event.metaKey;
      if (modifier && event.key.toLowerCase() === 's') {
        event.preventDefault(); void save(event.shiftKey); return;
      }
      // Undo/redo run even from a text field, but never touch server-side state.
      if (modifier && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (event.shiftKey) redo(); else undo();
        return;
      }
      // Never hijack keys while the user is typing.
      if (editing || event.altKey || modifier) return;
      if (event.key === 'ArrowRight' || event.key.toLowerCase() === 'n') { event.preventDefault(); move(1); }
      else if (event.key === 'ArrowLeft' || event.key.toLowerCase() === 'p') { event.preventDefault(); move(-1); }
      else if (event.key.toLowerCase() === 'v') { event.preventDefault(); void markVerified(); }
      else if (event.key.toLowerCase() === 'f') { event.preventDefault(); canvasApi.current?.fit(); }
      else if (event.key === '=' || event.key === '+') { event.preventDefault(); canvasApi.current?.zoomBy(1.25); }
      else if (event.key === '-' || event.key === '_') { event.preventDefault(); canvasApi.current?.zoomBy(1 / 1.25); }
      else if (event.key.toLowerCase() === 'd') { event.preventDefault(); setMode((m) => (m === 'draw' ? 'select' : 'draw')); }
      else if (event.key === 'Escape') { event.preventDefault(); setSelectedBoxIndex(null); }
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
  }, [section, verifyView, current?.id, boxes.length, selectedBoxIndex, save, markVerified, move, removeBox, undo, redo]);

  const renderTrainingSection = (type: 'detector' | 'ocr', trained: TrainingJob | null) => {
    const isOcr = type === 'ocr';
    const isActive = trained?.status === 'queued' || trained?.status === 'running';
    const otherTrainingActive = [detectorTrainingJob, ocrTrainingJob].some((item) => item?.model_type !== type && (item?.status === 'queued' || item?.status === 'running'));
    const units = isOcr ? 'plate crops' : 'images';
    // Only one split runs at a time. The backend resets a split's counter only
    // when that split actually starts, so a finished counter otherwise sits at
    // 100% for the rest of the epoch and looks like it is stuck.
    const valTotal = trained?.total_val || trained?.split_counts.val || 0;
    const testTotal = trained?.total_test || trained?.split_counts.test || 0;
    const valDone = trained?.current_val || 0;
    const testDone = trained?.current_test || 0;
    const batchesTotal = trained?.total_batches || 0;
    const trainingNow = !!isActive && batchesTotal > 0 && (trained?.current_batch || 0) < batchesTotal;
    const validatingNow = !!isActive && !trainingNow && valTotal > 0 && valDone < valTotal;
    const phase = !isActive ? (trained?.status || 'idle')
      : trainingNow ? 'Training'
        : validatingNow ? 'Validating'
          : 'Between epochs — saving metrics';
    return <section className={`training-page ${isOcr ? 'ocr-training-page' : 'detector-training-page'}`}>
      <header className="training-page-head">
        <div><h2>{isOcr ? 'OCR training' : 'Detection training'}</h2>
          <p>{isOcr ? 'Train character recognition from cropped license plates.' : 'Train plate localization: find the plate position in each image.'}</p></div>
        {trained?.model_ready && !isOcr && <button onClick={() => void activateModel(trained)} disabled={busy}>Activate detector model</button>}
        {trained?.model_ready && isOcr && <span className="training-source-note">Not activatable: this CRNN checkpoint cannot be used by the PaddleOCR recognition path.</span>}
      </header>
      <form className="training-config" onSubmit={startTraining}>
        <label>Dataset snapshot<select value={selectedSnapshot[type]} onChange={(e) => setSelectedSnapshot((old) => ({ ...old, [type]: e.target.value }))} required>
          <option value="">Create a snapshot first</option>{snapshots.filter((item) => item.task === type).map((item) => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.split_counts.train || 0}/{item.split_counts.val || 0}/{item.split_counts.test || 0} train/val/test</option>)}
        </select></label>
        <button type="button" onClick={() => void createSnapshot(type)} disabled={busy}>Create snapshot</button>
        <label>Epochs<input type="number" min="1" max="300" value={trainingConfig.epochs} onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: Number(e.target.value) })} /></label>
        <label>Batch size<input type="number" min="1" max="64" value={trainingConfig.batch_size} onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: Number(e.target.value) })} /></label>
        <label title="Background processes that load and prepare batches in parallel. More workers can speed up training, but each uses extra memory; 0 loads everything in the main process.">Workers<input type="number" min="0" max="16" value={trainingConfig.workers} onChange={(e) => setTrainingConfig({ ...trainingConfig, workers: Math.max(0, Number(e.target.value)) })} /></label>
        {!isOcr && <label>Image size<input type="number" min="320" max="1280" step="32" value={trainingConfig.image_size} onChange={(e) => setTrainingConfig({ ...trainingConfig, image_size: Number(e.target.value) })} /></label>}
        <button className="primary" type="submit" disabled={busy || isActive || otherTrainingActive || !selectedSnapshot[type]}>Start {isOcr ? 'OCR' : 'detector'} training</button>
      </form>
      {otherTrainingActive && <p className="training-source-note">Another training run is active. Wait for it to finish before starting this one.</p>}
      <p className="training-source-note">{isOcr
        ? 'Snapshot source: verified OCR text from imported OCR data and verified plate annotations. Crops from one source image stay in the same split.'
        : 'Snapshot source: verified plate boxes in the selected immutable detector dataset snapshot.'}</p>
      {isOcr && <p className="training-source-note ocr-engine-note">
        <strong>Engine: CRNN</strong> (CNN + BiLSTM + CTC, alphabet 0-9A-Z, 12 characters max). Recognition uses <strong>PaddleOCR</strong> PP-OCRv6-tiny,
        which cannot load these weights, so a finished CRNN run cannot be activated or selected for inference. It is kept for studying the training loop only.
      </p>}
      {isOcr && <OCRDatasetReview />}
      {trained ? <section className="training-status" aria-live="polite">
        <div className="training-status-head"><strong>{isOcr ? 'OCR' : 'Detector'} · {trained.status}</strong>
          {isActive && <span className="training-phase" aria-label="Current stage">Now: {phase}</span>}
          {isActive && <button className="stop-training" onClick={() => void cancelTraining(trained)} disabled={busy || trained.cancel_requested}>{trained.cancel_requested ? 'Stopping…' : 'Stop training'}</button>}
        </div>
        {trained.split_counts && Object.keys(trained.split_counts).length > 0 && <div className="training-splits">
          <div className={`split-row ${trainingNow ? 'is-running' : isActive ? 'is-idle' : ''}`}>
            <label>Train · epoch {trained.current_epoch} / {trained.total_epochs} · batch {trained.total_batches ? `${trained.current_batch} / ${trained.total_batches}` : 'preparing'} · {Math.min(trained.split_counts.train || 0, trained.current_batch * trained.config.batch_size).toLocaleString()} / {(trained.split_counts.train || 0).toLocaleString()} {units} this epoch
              <progress value={Math.min(trained.split_counts.train || 0, trained.current_batch * trained.config.batch_size)} max={trained.split_counts.train || 1} />
            </label>
            <span className="split-state">{trainingNow ? 'running' : !isActive ? 'finished' : 'not running now'}</span>
          </div>
          <div className={`split-row ${validatingNow ? 'is-running' : valDone >= valTotal && valTotal > 0 ? 'is-done' : 'is-idle'}`}>
            <label>Validation · {validatingNow ? `${valDone.toLocaleString()} / ${valTotal.toLocaleString()} ${units}` : valDone >= valTotal && valTotal > 0 ? `${valTotal.toLocaleString()} ${units} checked at the end of the last epoch` : `runs after each training epoch · ${valTotal.toLocaleString()} ${units} queued`}
              <progress value={valDone} max={valTotal || 1} />
            </label>
            <span className="split-state">{validatingNow ? 'running' : valDone >= valTotal && valTotal > 0 ? 'complete' : 'idle'}</span>
          </div>
          <div className={`split-row ${testDone >= testTotal && testTotal > 0 ? 'is-done' : 'is-idle'}`}>
            <label>Test · {testDone >= testTotal && testTotal > 0 ? `${testTotal.toLocaleString()} ${units} checked at the end of training` : `runs once, after training finishes · ${testTotal.toLocaleString()} ${units} queued`}
              <progress value={testDone} max={testTotal || 1} />
            </label>
            <span className="split-state">{testDone >= testTotal && testTotal > 0 ? 'complete' : 'idle'}</span>
          </div>
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
    <div className="app-body">
    {section === 'production' && <>
    <div className="production-heading"><div><h2>Verify Datasets</h2><p>Detection and OCR need different ground truth, so they are verified separately.</p></div>
      <div className="snapshot-actions">
        <button onClick={() => void syncDatasetsFolder()} disabled={busy || syncing}>{syncing ? 'Scanning…' : 'Sync datasets/imports/'}</button>
        <button onClick={() => void createSnapshot('detector')} disabled={busy}>Build detector snapshot</button>
        <button onClick={() => void createSnapshot('ocr')} disabled={busy}>Build OCR snapshot</button>
        <button onClick={() => void detect(true)} disabled={busy || batchActive || anyTrainingActive}>Detect all images</button>
      </div>
    </div>
    <section className="snapshot-list"><strong>Immutable snapshots</strong>{snapshots.length ? snapshots.map((item) => <span key={item.id}><b>{item.task}</b> · {item.id.slice(0, 8)} · {item.split_counts.train}/{item.split_counts.val}/{item.split_counts.test} train/val/test · {item.manifest_sha256.slice(0, 12)} <button className="delete-snapshot" onClick={() => void deleteSnapshot(item)} title="Delete snapshot">×</button></span>) : <span>No snapshots created yet.</span>}</section>
    {datasetMetrics && <section className="metrics-dashboard dataset-dashboard"><div className="metric"><span>Total images</span><strong>{datasetMetrics.total_images.toLocaleString()}</strong></div><div className="metric"><span>Verified plates</span><strong>{datasetMetrics.verified_plates.toLocaleString()}</strong></div><div className="metric"><span>Needs review</span><strong>{datasetMetrics.images_needing_review.toLocaleString()}</strong></div><div className="metric"><span>OCR exact match</span><strong>{percent(datasetMetrics.ocr_exact_match_accuracy)}</strong></div><div className="metric"><span>Character error rate</span><strong>{percent(datasetMetrics.character_error_rate)}</strong></div><div className="metric"><span>Detection recall</span><strong>{percent(datasetMetrics.detection_recall)}</strong></div><button className="metrics-refresh" onClick={() => void refreshDatasetMetrics()} disabled={busy}>Refresh statistics</button></section>}
    <nav className="verify-views" aria-label="Verification views">
      {([['detection', 'Detection', 'Boxes and plate locations for YOLO. OCR text is not required here.'],
         ['ocr', 'OCR Text', 'Exact characters on each plate for PaddleOCR recognition.'],
         ['samples', 'All Samples', 'Detection and OCR readiness side by side.']] as const).map(([id, label, hint]) =>
        <button type="button" key={id} className={verifyView === id ? 'active' : ''} onClick={() => setVerifyView(id)} title={hint}>{label}</button>)}
    </nav>

    {verifyView !== 'detection' && <>
      {verifyView === 'ocr' && <OCRTextReview source={source} onOpenDetection={(id) => { void openImageInDetection(id); setVerifyView('detection'); }} />}
      {verifyView === 'samples' && <AllSamples source={source} onOpen={(id) => { void openImageInDetection(id); setVerifyView('detection'); }} />}
    </>}

    {verifyView === 'detection' && <>
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
      <div className="canvas-toolbar" role="toolbar" aria-label="Canvas tools">
        <div className="mode-toggle" role="group" aria-label="Interaction mode">
          <button type="button" aria-pressed={mode === 'select'} className={mode === 'select' ? 'active' : ''}
            onClick={() => setMode('select')} title="Select mode: pick, move and resize existing boxes (V toggles)">Select</button>
          <button type="button" aria-pressed={mode === 'draw'} className={mode === 'draw' ? 'active' : ''}
            onClick={() => setMode('draw')} title="Draw mode: drag empty space to add a plate box (V toggles)">Draw</button>
        </div>
        <div className="zoom-controls" aria-label="Image zoom controls">
          <button aria-label="Zoom out" title="Zoom out (−)" onClick={() => canvasApi.current?.zoomBy(1 / 1.25)}>−</button>
          <button aria-label="Zoom in" title="Zoom in (+)" onClick={() => canvasApi.current?.zoomBy(1.25)}>+</button>
          <button onClick={() => canvasApi.current?.fit()} title="Fit image to screen (F)">Fit</button>
          <span aria-live="polite">{zoomLabel}%</span>
        </div>
        <div className="mode-toggle" role="group" aria-label="History">
          <button type="button" onClick={undo} disabled={!undoStack.length} title="Undo (Ctrl/⌘+Z)">Undo</button>
          <button type="button" onClick={redo} disabled={!redoStack.length} title="Redo (Ctrl/⌘+Shift+Z)">Redo</button>
        </div>
        {selectedBoxIndex != null && <button type="button" className="danger"
          onClick={() => { removeBox(selectedBoxIndex); setSelectedBoxIndex(null); }}
          title="Delete selected box (Delete)">Delete box</button>}
        <span className="shortcut-hint">Wheel zoom · Space+drag or middle-drag pan · V mode · F fit · Del remove</span>
      </div>
      {current && dirtyImageId === current.id && <span className="verify-badge pending">Unsaved edits</span>}
      <button className="primary" onClick={() => detect(false)} disabled={busy || jobActive || anyTrainingActive || !current}>Auto detect</button>
      <button className="primary" onClick={() => void save(false)} disabled={busy || !current || !hasUnsaved}>Save</button>
      <button className="primary" onClick={() => void save(true)} disabled={busy || !current || !hasUnsaved}>Save &amp; Next</button>
    </section>
    <section className="workspace">
      {current ? (imageEl
        ? <AnnotationCanvas ref={canvasApi} imageEl={imageEl} image={current} boxes={boxes} selectedIndex={selectedBoxIndex}
          mode={mode} disabled={busy} onZoomChange={setZoomLabel}
          onSelect={setSelectedBoxIndex} onEditStart={beginEdit} onChange={setBoxes} onCommit={commitBoxes} />
        : <div className="empty">Loading image…</div>)
        : <div className="empty">{source === 'datasets_folder' ? 'No images loaded. Click "Sync datasets/imports/" above.' : 'No registered images found. Import a dataset into datasets/raw/ first.'}</div>}
      <aside className="side-panel"><h2>Plate annotations <span>{boxes.length}</span></h2>
        <p className="hint">Drag on the image to add a plate. Click a box or a row to select it, then drag to move or pull a corner to resize.</p>
        <div className="annotation-list">{boxes.map((box, i) => <article key={i}
          className={`annotation ${selectedBoxIndex === i ? 'is-selected' : ''}`}
          onClick={() => setSelectedBoxIndex(i)} onContextMenu={(e) => { e.preventDefault(); removeBox(i); }}>
          <div className="annotation-head"><strong>Plate {i + 1}</strong>
            <span className={box.ai_detection ? 'ai-tag' : 'manual-tag'}>{box.ai_detection ? 'AI' : 'Manual'}</span>
            <button type="button" className="annotation-delete" aria-label={`Delete plate ${i + 1}`}
              title="Delete this box" onClick={(e) => { e.stopPropagation(); removeBox(i); }}>×</button>
          </div>
          <span className="annotation-coords">[{box.bbox_xyxy_pixels.map((v) => Math.round(v)).join(', ')}]</span>
          <PlateCrop image={imageEl} box={box.bbox_xyxy_pixels} />
          <label>Verified plate text<input ref={(element) => { ocrInputRefs.current[i] = element; }} onFocus={() => setSelectedBoxIndex(i)} value={box.verified_result} onChange={(e) => setText(i, e.target.value.toUpperCase())} placeholder={box.ai_result || 'Enter text'} /></label>
          {box.ai_detection && <div className="stats"><span>AI guess <b>{box.ai_result || '—'}</b></span><span>OCR confidence <b>{percent(box.ocr_confidence)}</b></span><span>Detection confidence <b>{percent(box.detection_confidence)}</b></span><span>Country <b>{box.country || 'Unknown'} · {percent(box.country_confidence)}</b></span></div>}
        </article>)}</div>
        {job && (job.status === 'running' || job.status === 'queued') && <div className="job"><div>{job.kind === 'batch' ? 'Batch detection' : 'Image detection'} {job.completed}/{job.total}</div><progress value={job.completed} max={job.total || 1} /></div>}
      </aside>
    </section>
    {message && <div className="status" role="status">{message}</div>}
    <footer>AI boxes are orange; manual boxes are green. <strong>OCR text is not required for detection training</strong> — a sample with verified boxes and no text is still detection-ready. Saving box or text edits returns the image to "Needs review" for both tasks, so verify it again afterwards. Predictions are never saved as ground truth automatically. Shortcuts: ←/→ or P/N navigate · Ctrl/⌘+S save · V verify · Delete remove selected box · T focus plate text.</footer>
    </>}
    </>}
    {section === 'detector' && renderTrainingSection('detector', detectorTrainingJob)}
    {section === 'ocr' && renderTrainingSection('ocr', ocrTrainingJob)}
    {section === 'recognition' && <RecognitionTest />}
    {message && section !== 'production' && <div className="status" role="status">{message}</div>}
    </div>
  </main>;
}
