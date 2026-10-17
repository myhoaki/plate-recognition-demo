import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';

type Model = { job_id: string; model_type: 'detector' | 'ocr'; created_at: string | null; best_epoch: number | null; engine?: string; usable?: boolean };
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

/** Short definitions so the pipeline terms are clear without leaving the page. */
const GLOSSARY: Record<string, string> = {
  'Bounding box': 'The rectangle YOLO drew around a plate, as pixel coordinates (x1, y1) top-left to (x2, y2) bottom-right.',
  'Detector confidence': 'How sure YOLO is that a box contains a plate. It says nothing about the characters.',
  'OCR confidence': 'How sure the recognizer is about the characters it read. It says nothing about the box.',
  'Checkpoint': 'The saved weights file a training run produced. Picking one pins the exact model used for a run.',
  'mAP50': 'Mean average precision at IoU 0.50 — detection quality averaged over confidence thresholds.',
  'mAP50–95': 'Stricter mAP averaged from IoU 0.50 to 0.95. Penalises loose boxes much more than mAP50.',
  'CER': 'Character error rate — roughly how many characters were wrong, as a fraction of the true text.',
};

function Term({ children }: { children: string }) {
  return <span className="term" tabIndex={0} title={GLOSSARY[children] || children}>{children}<sup>?</sup></span>;
}

const PIPELINE = [
  { id: 'image', label: 'Vehicle image', detail: 'Your photo, decoded to a pixel array.' },
  { id: 'detect', label: 'YOLO detection', detail: 'YOLO26n finds every plate and a confidence for each box.' },
  { id: 'crop', label: 'Plate crop', detail: 'A plain slice of the original pixels inside each box — no model involved.' },
  { id: 'ocr', label: 'OCR recognition', detail: 'PaddleOCR reads the characters from that crop.' },
  { id: 'result', label: 'Final result', detail: 'Boxes, text and both confidences reported together.' },
];

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
  const [error, setError] = useState('');
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    Promise.all([request<{ models: Model[] }>('/api/models'), request<{ snapshots: Snapshot[] }>('/api/dataset-snapshots?task=detector')])
      .then(([modelData, snapshotData]) => { setModels(modelData.models); setSnapshots(snapshotData.snapshots); setSnapshotId(snapshotData.snapshots[0]?.id || ''); })
      .catch((e: Error) => setMessage(e.message));
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

  const detectors = models.filter((model) => model.model_type === 'detector');
  const ocrModels = models.filter((model) => model.model_type === 'ocr');
  // `usable` comes from the server: it is false for checkpoints the pipeline
  // cannot actually run (currently every CRNN OCR run, because inference is
  // PaddleOCR). Those must not be offered as selectable recognisers.
  const usableOcrModels = ocrModels.filter((model) => model.usable !== false);
  const blockedOcrModels = ocrModels.filter((model) => model.usable === false);
  const selectedDetector = detectors.find((model) => model.job_id === detector) || null;
  const selectedOcrModel = ocrModels.find((model) => model.job_id === ocr) || null;
  const selectedOcrIsBlocked = !!selectedOcrModel && selectedOcrModel.usable === false;

  // A selection that the server would reject must not survive a re-fetch.
  useEffect(() => {
    if (selectedOcrIsBlocked) { setOcr(''); setResult(null); }
  }, [selectedOcrIsBlocked]);

  const run = async () => {
    if (!file) return;
    setBusy(true); setMessage('Running recognition…'); setResult(null); setError(''); setSelected(0);
    try {
      const form = new FormData(); form.append('file', file);
      const image = await request<{ id: string }>('/api/images/upload', { method: 'POST', body: form });
      const output = await request<Result>('/api/recognition', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: image.id, detector_job_id: detector || null, ocr_job_id: ocr || null }) });
      setResult(output);
      setMessage(`Completed in ${output.elapsed_ms} ms · ${output.detections.length} plate${output.detections.length === 1 ? '' : 's'} found.`);
    } catch (e) { setError((e as Error).message); setMessage(''); }
    finally { setBusy(false); }
  };

  const startEvaluation = async () => {
    if (!snapshotId) return;
    setBusy(true); setMessage('Queueing test split evaluation…');
    try {
      const state = await request<Evaluation>('/api/evaluations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dataset_snapshot_id: snapshotId, detector_job_id: detector || null, ocr_job_id: ocr || null }) });
      setEvaluation(state); setMessage('Evaluation queued. It runs on the server and uses the selected snapshot test split.');
    } catch (e) { setMessage((e as Error).message); }
    finally { setBusy(false); }
  };

  const metric = (key: string) => typeof evaluation?.results[key] === 'number' ? `${(Number(evaluation.results[key]) * 100).toFixed(1)}%` : 'Unavailable';
  const incorrect = Array.isArray(evaluation?.results.incorrect_examples) ? evaluation.results.incorrect_examples as Array<{ sample_id: string; filename: string; ground_truth: string[]; predictions: Array<{ text: string }> }> : [];
  const active = result && result.detections.length ? result.detections[Math.min(selected, result.detections.length - 1)] : null;
  const stageState = (id: string) => {
    if (busy) return id === 'image' ? 'active' : 'pending';
    if (!result) return 'pending';
    if (id === 'image' || id === 'detect') return 'done';
    if (id === 'crop' || id === 'ocr') return result.detections.length ? 'done' : 'skipped';
    return 'done';
  };

  return <section className="recognition-page">
    <header className="training-page-head"><div><h2>Recognition / Test</h2><p>Detection, cropping, and recognition run as separate stages. Follow the pipeline below to see exactly which pixels each stage saw.</p></div></header>

    <ol className="pipeline" aria-label="Pipeline stages">
      {PIPELINE.map((stage, index) => <li key={stage.id} className={`pipeline-stage stage-${stageState(stage.id)}`}>
        <span className="pipeline-index">{stageState(stage.id) === 'done' ? '✓' : index + 1}</span>
        <span className="pipeline-label">{stage.label}</span>
        <span className="pipeline-detail">{stage.detail}</span>
      </li>)}
    </ol>

    <div className="recognition-controls">
      <label>Vehicle image<input type="file" accept="image/jpeg,image/png,image/webp,image/bmp" onChange={(event) => { setFile(event.target.files?.[0] || null); setResult(null); setError(''); }} /></label>
      <label>Detector <Term>checkpoint</Term><select value={detector} onChange={(event) => { setDetector(event.target.value); setResult(null); }}>
        <option value="">Active YOLO26 detector (active_detector.pt)</option>
        {detectors.map((model) => <option key={model.job_id} value={model.job_id}>Run {model.job_id.slice(0, 8)} · epoch {model.best_epoch ?? '—'}</option>)}
      </select></label>
      <label>Recogniser<select value={ocr} onChange={(event) => { setOcr(event.target.value); setResult(null); }}>
        <option value="">PaddleOCR PP-OCRv6-tiny (default)</option>
        {usableOcrModels.map((model) => <option key={model.job_id} value={model.job_id}>{model.engine ?? 'PaddleOCR'} run {model.job_id.slice(0, 8)} · epoch {model.best_epoch ?? '—'}</option>)}
        {blockedOcrModels.length > 0 && <optgroup label={`CRNN runs — incompatible with PaddleOCR (${blockedOcrModels.length})`}>
          {blockedOcrModels.map((model) => <option key={model.job_id} value={model.job_id} disabled>{model.engine?.toUpperCase() ?? 'CRNN'} run {model.job_id.slice(0, 8)} · epoch {model.best_epoch ?? '—'} · not selectable</option>)}
        </optgroup>}
      </select></label>
      <button className="primary" disabled={!file || busy} onClick={() => void run()}>{busy ? 'Recognizing…' : 'Run recognition'}</button>
    </div>

    <section className="engine-panel" aria-label="Models actually used">
      <div><small>Detector in use</small><strong>{selectedDetector ? `Run ${selectedDetector.job_id.slice(0, 8)} (epoch ${selectedDetector.best_epoch ?? '—'})` : 'Active YOLO26n · active_detector.pt'}</strong></div>
      <div><small>Recogniser in use</small><strong>{selectedOcrModel && !selectedOcrIsBlocked ? `${selectedOcrModel.engine ?? 'PaddleOCR'} run ${selectedOcrModel.job_id.slice(0, 8)}` : 'PaddleOCR PP-OCRv6-tiny'}</strong></div>
      {blockedOcrModels.length > 0 && <p className="engine-warning" role="note">
        <strong>{blockedOcrModels.length} CRNN training {blockedOcrModels.length === 1 ? 'run is' : 'runs are'} shown but not selectable.</strong>{' '}
        The OCR trainer produces a <strong>CRNN</strong> checkpoint, but recognition runs <strong>PaddleOCR</strong>. Different architectures cannot be
        interchanged, so those runs are excluded here and the API rejects them if one is submitted. Training a PaddleOCR recogniser is not implemented yet.
      </p>}
    </section>

    {message && <p className="recognition-message" role="status">{message}</p>}
    {error && <p className="recognition-error" role="alert">{error}</p>}

    {result && <div className="recognition-result">
      <div className="recognition-original">
        {/* The overlay boxes are positioned in % of their containing block, so the
            image needs its own wrapper sized to the rendered image box. */}
        <div className="recognition-canvas" style={{ '--plate-aspect': `${result.width} / ${result.height}` } as CSSProperties}>
          <img src={`${result.file_url}?v=${result.image_id}`} alt={result.filename} />
          {result.detections.map((detection, index) => { const [x1, y1, x2, y2] = detection.bbox_xyxy_pixels; return <div key={index} onClick={() => setSelected(index)} className={`recognition-box ${selected === index ? 'is-selected' : ''}`} style={{ left: `${x1 / result.width * 100}%`, top: `${y1 / result.height * 100}%`, width: `${(x2 - x1) / result.width * 100}%`, height: `${(y2 - y1) / result.height * 100}%` }} aria-label={`Plate ${index + 1}: ${detection.text || 'no text'}`}><span>{index + 1}</span></div>; })}
        </div>
      </div>

      <div className="recognition-detections">
        <h3>Detected plates <span>{result.detections.length}</span></h3>
        {!result.detections.length && <p className="empty">No plates detected. The detector found nothing above its confidence threshold, so there is no crop and no OCR step.</p>}
        {result.detections.map((detection, index) => <article className={`recognition-detection ${selected === index ? 'is-selected' : ''}`} key={index} onClick={() => setSelected(index)}>
          <header><strong>Plate {index + 1}</strong>{detection.crop_data_url && <img className="plate-crop" src={detection.crop_data_url} alt={`Plate crop ${index + 1}`} />}</header>
          <span className="recognition-text">{detection.text || <em>No text recognised</em>}</span>
          <span className="confidence confidence-detector"><Term>Detector confidence</Term> · {(detection.detection_confidence * 100).toFixed(1)}%</span>
          <span className="confidence confidence-ocr"><Term>OCR confidence</Term> · {detection.ocr_confidence == null ? 'Not provided' : `${(detection.ocr_confidence * 100).toFixed(1)}%`}</span>
        </article>)}

        {active && <section className="plate-inspector" aria-label="Plate inspector">
          <h4>Plate {Math.min(selected, result.detections.length - 1) + 1} details</h4>
          <dl>
            <dt><Term>Bounding box</Term></dt><dd>{active.bbox_xyxy_pixels.map((value) => Math.round(value)).join(', ')}</dd>
            <dt>Box size</dt><dd>{Math.round(active.bbox_xyxy_pixels[2] - active.bbox_xyxy_pixels[0])} × {Math.round(active.bbox_xyxy_pixels[3] - active.bbox_xyxy_pixels[1])} px</dd>
            <dt><Term>Detector confidence</Term></dt><dd>{(active.detection_confidence * 100).toFixed(1)}%</dd>
            <dt><Term>OCR confidence</Term></dt><dd>{active.ocr_confidence == null ? 'Not provided' : `${(active.ocr_confidence * 100).toFixed(1)}%`}</dd>
            <dt>Recognised text</dt><dd>{active.text || <em>none</em>}</dd>
          </dl>
          <p className="inspector-note">The two confidences measure different things and are never combined: detector confidence scores the box, OCR confidence scores the characters.</p>
        </section>}

        <p className="elapsed-note">Total pipeline time {result.elapsed_ms} ms (detection + crop + OCR). The backend reports one combined figure; per-stage timings are not available yet.</p>
      </div>
    </div>}

    <section className="evaluation-panel"><header><div><h3>Batch evaluation</h3><p>Evaluate the frozen test split. Detection boxes use one-to-one IoU ≥ 0.50 matching; plate text is trimmed and case-insensitive, with all other characters preserved.</p></div><button className="primary" onClick={() => void startEvaluation()} disabled={!snapshotId || busy || !!evaluation && ['queued', 'running'].includes(evaluation.status)}>Evaluate test split</button></header>
      <div className="evaluation-controls"><label>Detector test snapshot<select value={snapshotId} onChange={(event) => setSnapshotId(event.target.value)}><option value="">No detector snapshot</option>{snapshots.map((snapshot) => <option key={snapshot.id} value={snapshot.id}>{snapshot.id.slice(0, 8)} · {snapshot.split_counts.test || 0} test images</option>)}</select></label></div>
      {evaluation && <><p className="recognition-message">Evaluation · {evaluation.status} · {evaluation.processed.toLocaleString()} / {evaluation.total.toLocaleString()}{evaluation.error ? ` · ${evaluation.error}` : ''}</p>{['queued', 'running'].includes(evaluation.status) && <progress className="evaluation-progress" value={evaluation.processed} max={evaluation.total || 1} />}
        {evaluation.status === 'completed' && <><div className="training-metric-grid">
          <div><small>Detection precision</small><b>{metric('detection_precision')}</b></div>
          <div><small>Detection recall</small><b>{metric('detection_recall')}</b></div>
          <div><small><Term>mAP50</Term></small><b>{metric('detection_map50')}</b></div>
          <div><small><Term>mAP50–95</Term></small><b>{metric('detection_map50_95')}</b></div>
          <div><small>OCR exact string</small><b>{metric('ocr_exact_accuracy')}</b></div>
          <div><small><Term>CER</Term></small><b>{metric('character_error_rate')}</b></div>
          <div><small>End-to-end accuracy</small><b>{metric('end_to_end_accuracy')}</b></div>
        </div>
          {evaluation.results.notes && <p className="recognition-message">{String(evaluation.results.notes)}</p>}
          <p>Missed plates: {String(evaluation.results.missed_plates ?? 0)} · False detections: {String(evaluation.results.false_detections ?? 0)} · OCR samples: {String(evaluation.results.ocr_evaluated_plates ?? 0)}</p>
          {incorrect.length > 0 && <div className="evaluation-errors"><h4>Incorrect predictions ({incorrect.length} shown)</h4>{incorrect.map((row) => <article key={row.sample_id}><span>{row.filename}</span><span>Expected: {row.ground_truth.join(', ') || 'No plate'}</span><span>Predicted: {row.predictions.map((item) => item.text || '—').join(', ') || 'No detections'}</span></article>)}</div>}
        </>}
      </>}
    </section>
  </section>;
}