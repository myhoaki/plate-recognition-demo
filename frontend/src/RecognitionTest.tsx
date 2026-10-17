import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';

type Model = { job_id: string; model_type: 'detector' | 'ocr'; created_at: string | null; best_epoch: number | null };
type Detection = { bbox_xyxy_pixels: number[]; text: string; detection_confidence: number; ocr_confidence: number | null; crop_data_url?: string };
type Result = { image_id: string; filename: string; width: number; height: number; file_url: string; detections: Detection[]; elapsed_ms: number };
type Snapshot = { id: string; task: 'detector' | 'ocr'; split_counts: Record<string, number> };
type Evaluation = { id: string; status: string; processed: number; total: number; error?: string | null; results: Record<string, unknown> };

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

export default function RecognitionTest() {
  const [models, setModels] = useState<Model[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [snapshotId, setSnapshotId] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [detector, setDetector] = useState('');
  const [ocr, setOcr] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);

  useEffect(() => {
    Promise.all([request<{ models: Model[] }>('/api/models'), request<{ snapshots: Snapshot[] }>('/api/dataset-snapshots?task=detector')])
      .then(([modelData, snapshotData]) => { setModels(modelData.models); setSnapshots(snapshotData.snapshots); setSnapshotId(snapshotData.snapshots[0]?.id || ''); })
      .catch((error) => setMessage(error.message));
  }, []);

  useEffect(() => {
    if (!evaluation || !['queued', 'running'].includes(evaluation.status)) return;
    let cancelled = false;
    const poll = async () => {
      try { const state = await request<Evaluation>(`/api/evaluations/${evaluation.id}`); if (!cancelled) setEvaluation(state); }
      catch (error) { if (!cancelled) setMessage((error as Error).message); }
    };
    const timer = window.setInterval(poll, 1500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [evaluation?.id, evaluation?.status]);

  const run = async () => {
    if (!file) return;
    setBusy(true); setMessage('Running recognition…'); setResult(null);
    try {
      const form = new FormData(); form.append('file', file);
      const image = await request<{ id: string }>('/api/images/upload', { method: 'POST', body: form });
      const output = await request<Result>('/api/recognition', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: image.id, detector_job_id: detector || null, ocr_job_id: ocr || null }) });
      setResult(output); setMessage(`Completed in ${output.elapsed_ms} ms · ${output.detections.length} plate${output.detections.length === 1 ? '' : 's'} found.`);
    } catch (error) { setMessage((error as Error).message); }
    finally { setBusy(false); }
  };

  const detectors = models.filter((model) => model.model_type === 'detector');
  const ocrModels = models.filter((model) => model.model_type === 'ocr');
  const startEvaluation = async () => {
    if (!snapshotId) return;
    setBusy(true); setMessage('Queueing test split evaluation…');
    try {
      const state = await request<Evaluation>('/api/evaluations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dataset_snapshot_id: snapshotId, detector_job_id: detector || null, ocr_job_id: ocr || null }) });
      setEvaluation(state); setMessage('Evaluation queued. It runs on the server and uses the selected snapshot test split.');
    } catch (error) { setMessage((error as Error).message); }
    finally { setBusy(false); }
  };
  const metric = (key: string) => typeof evaluation?.results[key] === 'number' ? `${(Number(evaluation.results[key]) * 100).toFixed(1)}%` : 'Unavailable';
  const incorrect = Array.isArray(evaluation?.results.incorrect_examples) ? evaluation.results.incorrect_examples as Array<{ sample_id: string; filename: string; ground_truth: string[]; predictions: Array<{ text: string }> }> : [];
  return <section className="recognition-page">
    <header className="training-page-head"><div><h2>Recognition / Test</h2><p>Detection, cropping, and recognition run as separate stages: YOLO26 finds plates, OpenCV crops them, and PaddleOCR reads the characters.</p></div></header>
    <div className="recognition-controls">
      <label>Vehicle image<input type="file" accept="image/jpeg,image/png,image/webp,image/bmp" onChange={(event) => { setFile(event.target.files?.[0] || null); setResult(null); }} /></label>
      <label>Detector checkpoint<select value={detector} onChange={(event) => setDetector(event.target.value)}><option value="">Active YOLO26 detector</option>{detectors.map((model) => <option key={model.job_id} value={model.job_id}>Run {model.job_id.slice(0, 8)} · epoch {model.best_epoch ?? '—'}</option>)}</select></label>
      <label>OCR checkpoint<select value={ocr} onChange={(event) => setOcr(event.target.value)}><option value="">PaddleOCR PP-OCRv6</option>{ocrModels.map((model) => <option key={model.job_id} value={model.job_id}>Run {model.job_id.slice(0, 8)} · epoch {model.best_epoch ?? '—'}</option>)}</select></label>
      <button className="primary" disabled={!file || busy} onClick={() => void run()}>{busy ? 'Recognizing…' : 'Run recognition'}</button>
    </div>
    {message && <p className="recognition-message" role="status">{message}</p>}
    {result && <div className="recognition-result">
      {/* The overlay boxes are positioned in % of their containing block, so the
          image needs its own wrapper. Otherwise they resolve against the grid
          column instead of the rendered image and drift off the plates. */}
      <div className="recognition-original"><div className="recognition-canvas" style={{ '--plate-aspect': `${result.width} / ${result.height}` } as CSSProperties}><img src={`${result.file_url}?v=${result.image_id}`} alt={result.filename} />{result.detections.map((detection, index) => { const [x1,y1,x2,y2] = detection.bbox_xyxy_pixels; return <div key={index} className="recognition-box" style={{ left: `${x1 / result.width * 100}%`, top: `${y1 / result.height * 100}%`, width: `${(x2-x1) / result.width * 100}%`, height: `${(y2-y1) / result.height * 100}%` }} aria-label={`Plate ${index + 1}: ${detection.text}`} />; })}</div></div>
      <div className="recognition-detections"><h3>Detected plates</h3>{result.detections.length ? result.detections.map((detection, index) => <article className="recognition-detection" key={index}>
        {detection.crop_data_url && <img className="plate-crop" src={detection.crop_data_url} alt={`Plate crop ${index + 1}`} />}
        <strong>{detection.text || 'No text recognized'}</strong><span>Detector confidence · {(detection.detection_confidence * 100).toFixed(1)}%</span><span>OCR confidence · {detection.ocr_confidence == null ? 'Not provided' : `${(detection.ocr_confidence * 100).toFixed(1)}%`}</span>
      </article>) : <p>No plates detected.</p>}</div>
    </div>}
    <section className="evaluation-panel"><header><div><h3>Batch evaluation</h3><p>Evaluate the frozen test split. Detection boxes use one-to-one IoU ≥ 0.50 matching; plate text is trimmed and case-insensitive, with other characters preserved.</p></div><button className="primary" onClick={() => void startEvaluation()} disabled={!snapshotId || busy || !!evaluation && ['queued','running'].includes(evaluation.status)}>Evaluate test split</button></header>
      <div className="evaluation-controls"><label>Detector test snapshot<select value={snapshotId} onChange={(event) => setSnapshotId(event.target.value)}><option value="">No detector snapshot</option>{snapshots.map((snapshot) => <option key={snapshot.id} value={snapshot.id}>{snapshot.id.slice(0, 8)} · {snapshot.split_counts.test || 0} test images</option>)}</select></label></div>
      {evaluation && <><p className="recognition-message">Evaluation · {evaluation.status} · {evaluation.processed.toLocaleString()} / {evaluation.total.toLocaleString()}{evaluation.error ? ` · ${evaluation.error}` : ''}</p>{['queued','running'].includes(evaluation.status) && <progress className="evaluation-progress" value={evaluation.processed} max={evaluation.total || 1} />}
        {evaluation.status === 'completed' && <><div className="training-metric-grid"><div><small>Detection precision</small><b>{metric('detection_precision')}</b></div><div><small>Detection recall</small><b>{metric('detection_recall')}</b></div><div><small>mAP50</small><b>{metric('detection_map50')}</b></div><div><small>mAP50–95</small><b>{metric('detection_map50_95')}</b></div><div><small>OCR exact string</small><b>{metric('ocr_exact_accuracy')}</b></div><div><small>Character error rate</small><b>{metric('character_error_rate')}</b></div><div><small>End-to-end accuracy</small><b>{metric('end_to_end_accuracy')}</b></div></div>
          {evaluation.results.notes && <p className="recognition-message">{String(evaluation.results.notes)}</p>}
          <p>Missed plates: {String(evaluation.results.missed_plates ?? 0)} · False detections: {String(evaluation.results.false_detections ?? 0)} · OCR samples: {String(evaluation.results.ocr_evaluated_plates ?? 0)}</p>
          {incorrect.length > 0 && <div className="evaluation-errors"><h4>Incorrect predictions ({incorrect.length} shown)</h4>{incorrect.map((row) => <article key={row.sample_id}><span>{row.filename}</span><span>Expected: {row.ground_truth.join(', ') || 'No plate'}</span><span>Predicted: {row.predictions.map((item) => item.text || '—').join(', ') || 'No detections'}</span></article>)}</div>}</>}
      </>}
    </section>
  </section>;
}
