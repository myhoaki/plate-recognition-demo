import { Fragment, type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Image as KImage, Layer, Rect, Stage, Text } from 'react-konva';
import type Konva from 'konva';

type Annotation = {
  id?: number; bbox_xyxy_pixels: number[]; ai_detection: boolean; ai_result?: string | null;
  verified_result?: string | null; detection_confidence?: number | null; ocr_confidence?: number | null;
  country?: string | null; country_confidence?: number | null;
};
type ImageRecord = { id: string; filename: string; width: number; height: number; detection_complete: boolean; verified: boolean; verified_at?: string | null; file_url: string; annotations?: Annotation[] };
type Job = { job_id?: string; id?: string; kind?: 'single' | 'batch'; status: string; total: number; completed: number; detected: number; failed: number; error?: string | null; existing?: boolean };
type TrainingJob = { id: string; status: string; config: { epochs: number; batch_size: number; image_size: number; device: string }; split_counts: Record<string, number>; current_epoch: number; total_epochs: number; metrics: Record<string, number>; model_ready: boolean; error?: string | null };
type RefreshState = { running: boolean; processed: number; total: number; added_or_changed: number; error?: string | null };
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
  const [source, setSource] = useState<'rawimages' | 'upload'>('rawimages');
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [imageEl, setImageEl] = useState<HTMLImageElement | null>(null);
  const [stageSize, setStageSize] = useState({ width: 800, height: 520 });
  const [zoom, setZoom] = useState(1);
  const [busy, setBusy] = useState(false);
  const [dirtyImageId, setDirtyImageId] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [activeJobs, setActiveJobs] = useState<Job[]>([]);
  const [trainingJob, setTrainingJob] = useState<TrainingJob | null>(null);
  const [showTrainingForm, setShowTrainingForm] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({ epochs: 30, batch_size: 8, image_size: 640, device: 'auto' });
  const [drawing, setDrawing] = useState<number[] | null>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const current = images[index];
  const jobActive = job?.status === 'queued' || job?.status === 'running';
  const batchActive = activeJobs.some((item) => item.kind === 'batch');
  const trainingActive = trainingJob?.status === 'queued' || trainingJob?.status === 'running';

  useEffect(() => {
    let cancelled = false;
    api<{ job: TrainingJob | null }>('/api/training-jobs/latest').then((result) => {
      if (!cancelled && result.job) setTrainingJob(result.job);
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!trainingJob?.id || !trainingActive) return;
    let cancelled = false;
    let timer = 0;
    const poll = async () => {
      try {
        const state = await api<TrainingJob>(`/api/training-jobs/${trainingJob.id}`);
        if (cancelled) return;
        setTrainingJob(state);
        if (state.status === 'queued' || state.status === 'running') timer = window.setTimeout(poll, 2000);
        else setMessage(state.status === 'completed' ? 'Training completed. Review the test metrics, then activate the model if you want to use it.' : `Training ${state.status}: ${state.error || 'see training details.'}`);
      } catch { if (!cancelled) timer = window.setTimeout(poll, 4000); }
    };
    timer = window.setTimeout(poll, 1000);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [trainingJob?.id, trainingActive]);

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
    const data = await api<{ images: ImageRecord[]; total: number }>(`/api/images?source=${src}&limit=${PAGE_SIZE}&offset=${offset}`);
    setImages(data.images);
    setImageTotal(data.total);
    setPageOffset(offset);
    const nextIndex = preferredId ? data.images.findIndex((item) => item.id === preferredId) : -1;
    setIndex(nextIndex >= 0 ? nextIndex : 0);
  }, [source]);

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
        await loadImages(source, current?.id, pageOffset);
        if (current) {
          const detail = await api<ImageRecord>(`/api/images/${current.id}`);
          if (!cancelled) setBoxes((detail.annotations || []).map((a) => ({ ...a, verified_result: a.verified_result || '' })));
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

  useEffect(() => { loadImages().catch((e) => setMessage(e.message)); }, [loadImages]);
  useEffect(() => {
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
  const changeSource = async (value: 'rawimages' | 'upload') => { setSource(value); setIndex(0); setBoxes([]); setMessage(''); await loadImages(value, undefined, 0); };
  const upload = async (file?: File) => {
    if (!file) return;
    setBusy(true); setMessage('Uploading image…');
    try {
      const form = new FormData(); form.append('file', file);
      const created = await api<ImageRecord>('/api/images/upload', { method: 'POST', body: form });
      setSource('upload'); await loadImages('upload', created.id); setMessage('Image uploaded.');
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const refresh = async () => {
    setBusy(true); setMessage('Scanning rawimages/…');
    try {
      await api<RefreshState>('/api/images/refresh', { method: 'POST' });
      let result = await api<RefreshState>('/api/images/refresh');
      while (result.running) {
        setMessage(`Scanning rawimages/… ${result.processed.toLocaleString()} / ${result.total.toLocaleString()}`);
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
        result = await api<RefreshState>('/api/images/refresh');
      }
      if (result.error) throw new Error(result.error);
      await loadImages('rawimages', current?.id, pageOffset);
      setMessage(`Gallery refreshed. ${result.added_or_changed.toLocaleString()} new or changed images.`);
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
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
    setBusy(true); setMessage('Queueing detector training…');
    try {
      const job = await api<TrainingJob>('/api/training-jobs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingConfig),
      });
      setTrainingJob(job);
      setShowTrainingForm(false);
      setMessage(job.status === 'running' ? 'Detector training is already running.' : 'Detector training queued. It continues on the server if you close this page.');
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const activateModel = async () => {
    if (!trainingJob) return;
    setBusy(true);
    try {
      await api(`/api/training-jobs/${trainingJob.id}/activate`, { method: 'POST' });
      setMessage('Trained detector activated. New detections will use it with the existing OCR model.');
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const save = async () => {
    if (!current) return;
    setBusy(true); setMessage('Saving annotations…');
    try {
      await api(`/api/images/${current.id}/annotations`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ annotations: boxes.map((b) => ({ bbox_xyxy_pixels: b.bbox_xyxy_pixels, verified_result: b.verified_result })) }),
      });
      setMessage('Annotations saved to PostgreSQL.');
      setDirtyImageId(null);
      setImages((all) => all.map((item) => item.id === current.id ? { ...item, verified: false, verified_at: null } : item));
      const detail = await api<ImageRecord>(`/api/images/${current.id}`);
      setBoxes((detail.annotations || []).map((a) => ({ ...a, verified_result: a.verified_result || '' })));
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const markVerified = async () => {
    if (!current || dirtyImageId === current.id) return;
    const verified = !current.verified;
    setBusy(true);
    try {
      const result = await api<{ image: ImageRecord }>(`/api/images/${current.id}/verification`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ verified }),
      });
      setImages((all) => all.map((item) => item.id === current.id ? { ...item, verified: result.image.verified, verified_at: result.image.verified_at } : item));
      setMessage(verified ? 'Image marked verified.' : 'Verification removed.');
    } catch (e) { setMessage((e as Error).message); } finally { setBusy(false); }
  };
  const removeBox = (idx: number) => {
    setBoxes((all) => all.filter((_, n) => n !== idx));
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
      if (current) setDirtyImageId(current.id);
    }
    setDrawing(null);
  };
  const setText = (idx: number, text: string) => {
    setBoxes((all) => all.map((box, i) => i === idx ? { ...box, verified_result: text } : box));
    if (current) setDirtyImageId(current.id);
  };

  return <main className="app-shell">
    <header className="topbar"><div><h1>PlateLens</h1><p>License plate detection and annotation</p></div>
      <div className="top-actions"><label className="upload button">Upload image<input type="file" accept="image/*" onChange={(e) => upload(e.target.files?.[0])} /></label>
        <button onClick={() => setShowTrainingForm((value) => !value)} disabled={busy || trainingActive}>Train detector</button>
        <button onClick={() => detect(true)} disabled={busy || batchActive || trainingActive}>Detect all raw images</button></div>
    </header>
    {showTrainingForm && <section className="training-panel">
      <form onSubmit={startTraining}>
        <label>Epochs<input type="number" min="1" max="300" value={trainingConfig.epochs} onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: Number(e.target.value) })} /></label>
        <label>Batch size<input type="number" min="1" max="64" value={trainingConfig.batch_size} onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: Number(e.target.value) })} /></label>
        <label>Image size<input type="number" min="320" max="1280" step="32" value={trainingConfig.image_size} onChange={(e) => setTrainingConfig({ ...trainingConfig, image_size: Number(e.target.value) })} /></label>
        <span className="split-note">Verified images split: 80% train · 10% validation · 10% test</span>
        <button className="primary" type="submit" disabled={busy || trainingActive}>Start training</button>
      </form>
      <p>Only images marked Verified are included. At least 3 verified images are required.</p>
    </section>}
    {trainingJob && <section className="training-status" aria-live="polite">
      <div className="training-status-head"><strong>Detector training · {trainingJob.status}</strong>
        {trainingJob.model_ready && <button onClick={activateModel} disabled={busy}>Use this model</button>}
      </div>
      {trainingActive && <><span>Epoch {trainingJob.current_epoch} / {trainingJob.total_epochs} · batch {trainingJob.config.batch_size}</span><progress value={trainingJob.current_epoch} max={trainingJob.total_epochs || 1} /></>}
      {Object.keys(trainingJob.split_counts).length > 0 && <span>Train {trainingJob.split_counts.train} · Validation {trainingJob.split_counts.val} · Test {trainingJob.split_counts.test}</span>}
      {trainingJob.metrics['metrics/mAP50(B)'] != null && <span>{trainingActive ? 'Validation' : 'Test'} mAP50: {(trainingJob.metrics['metrics/mAP50(B)'] * 100).toFixed(1)}% · mAP50–95: {((trainingJob.metrics['metrics/mAP50-95(B)'] || 0) * 100).toFixed(1)}%</span>}
      {trainingJob.error && <span className="training-error">{trainingJob.error}</span>}
    </section>}
    {activeJobs.length > 0 && <section className="queue-status" aria-live="polite">
      <strong>Detection queue</strong>
      {activeJobs.map((item) => <div className="queue-item" key={item.id}>
        <span>{item.kind === 'batch' ? 'Batch' : 'Image'} · {item.status === 'queued' ? 'Waiting' : 'Running'}</span>
        <span>{item.completed.toLocaleString()} / {item.total.toLocaleString()} complete</span>
        <progress value={item.completed} max={item.total || 1} />
      </div>)}
    </section>}
    <section className="toolbar"><select value={source} onChange={(e) => changeSource(e.target.value as 'rawimages' | 'upload')}><option value="rawimages">Server rawimages</option><option value="upload">Uploads</option></select>
      {source === 'rawimages' && <button onClick={refresh} disabled={busy}>Refresh images</button>}
      <div className="image-review-controls" aria-label="Image review navigation">
        <button aria-label="Previous image" title="Previous image" onClick={() => move(-1)} disabled={!images.length || (!pageOffset && index === 0)}>←</button>
        <span className="counter">{images.length ? `${pageOffset + index + 1} / ${imageTotal}` : 'No images'}</span>
        <button aria-label="Next image" title="Next image" onClick={() => move(1)} disabled={!images.length || pageOffset + index >= imageTotal - 1}>→</button>
        {current && <button className={`verify-action ${current.verified ? 'verified' : ''}`} onClick={markVerified} disabled={busy || dirtyImageId === current.id}>
          {current.verified ? 'Undo verified' : 'Mark verified'}
        </button>}
      </div>
      <form className="jump-form" onSubmit={jumpToOrder}>
        <label htmlFor="jump-order">Go to image</label>
        <input id="jump-order" type="number" min="1" max={imageTotal || 1} value={jumpOrder} onChange={(e) => setJumpOrder(e.target.value)} disabled={!imageTotal || busy} />
        <button type="submit" disabled={!imageTotal || busy}>Go</button>
      </form>
      <span className="filename">{current?.filename || 'Add images to rawimages/ or upload one'}</span>
      <div className="zoom-controls" aria-label="Image zoom controls">
        <button aria-label="Zoom out" title="Zoom out" onClick={() => setZoom((value) => Math.max(0.25, Math.round((value - 0.25) * 100) / 100))} disabled={zoom <= 0.25}>−</button>
        <span>{Math.round(zoom * 100)}%</span>
        <button aria-label="Zoom in" title="Zoom in" onClick={() => setZoom((value) => Math.min(4, Math.round((value + 0.25) * 100) / 100))} disabled={zoom >= 4}>+</button>
        <button onClick={() => setZoom(1)} disabled={zoom === 1}>Fit</button>
      </div>
      {current && dirtyImageId === current.id && <span className="verify-badge pending">Unsaved edits</span>}
      <button className="primary" onClick={() => detect(false)} disabled={busy || jobActive || trainingActive || !current}>Auto detect</button>
      <button className="primary" onClick={save} disabled={busy || !current}>Save labels</button>
    </section>
    <section className="workspace">
      <div className="canvas-wrap">
        {current && imageEl ? <Stage ref={stageRef} width={stageSize.width * zoom} height={stageSize.height * zoom} onMouseDown={startDraw} onMouseMove={updateDraw} onMouseUp={endDraw} onTouchStart={startDraw as never} onTouchMove={updateDraw} onTouchEnd={endDraw}>
          <Layer><KImage image={imageEl} width={current.width * scale} height={current.height * scale} listening={false} />
            {boxes.map((box, i) => { const [x1,y1,x2,y2] = box.bbox_xyxy_pixels; return <Fragment key={i}>
              <Rect x={x1*scale} y={y1*scale} width={(x2-x1)*scale} height={(y2-y1)*scale} stroke={box.ai_detection ? '#f59e0b' : '#16a34a'} strokeWidth={2} onContextMenu={(e) => { e.evt.preventDefault(); removeBox(i); }} />
              <Text x={x1*scale} y={Math.max(0,y1*scale-21)} text={box.ai_result || box.verified_result || `Plate ${i+1}`} fontSize={14} fill="white" padding={3} listening={false} />
            </Fragment>; })}
            {drawing && <Rect x={Math.min(drawing[0],drawing[2])*scale} y={Math.min(drawing[1],drawing[3])*scale} width={Math.abs(drawing[2]-drawing[0])*scale} height={Math.abs(drawing[3]-drawing[1])*scale} stroke="#38bdf8" dash={[5,3]} />}
          </Layer>
        </Stage> : <div className="empty">{current ? 'Loading image…' : 'No images found. Put images in rawimages/ or upload one.'}</div>}
      </div>
      <aside className="side-panel"><h2>Plate annotations <span>{boxes.length}</span></h2><p className="hint">Draw a box on the image. Right-click a box row to remove it.</p>
        <div className="annotation-list">{boxes.map((box, i) => <article className="annotation" key={i} onContextMenu={(e) => { e.preventDefault(); removeBox(i); }}>
          <div className="annotation-head"><strong>Plate {i + 1}</strong><span className={box.ai_detection ? 'ai-tag' : 'manual-tag'}>{box.ai_detection ? 'AI' : 'Manual'}</span></div>
          <PlateCrop image={imageEl} box={box.bbox_xyxy_pixels} />
          <label>Verified plate text<input value={box.verified_result} onChange={(e) => setText(i, e.target.value.toUpperCase())} placeholder={box.ai_result || 'Enter text'} /></label>
          {box.ai_detection && <div className="stats"><span>AI guess <b>{box.ai_result || '—'}</b></span><span>OCR confidence <b>{percent(box.ocr_confidence)}</b></span><span>Detection confidence <b>{percent(box.detection_confidence)}</b></span><span>Country <b>{box.country || 'Unknown'} · {percent(box.country_confidence)}</b></span></div>}
        </article>)}</div>
        {job && (job.status === 'running' || job.status === 'queued') && <div className="job"><div>{job.kind === 'batch' ? 'Batch detection' : 'Image detection'} {job.completed}/{job.total}</div><progress value={job.completed} max={job.total || 1} /></div>}
      </aside>
    </section>
    {message && <div className="status" role="status">{message}</div>}
    <footer>AI boxes are orange; manual boxes are green. Drag on the image to add a box.</footer>
  </main>;
}
