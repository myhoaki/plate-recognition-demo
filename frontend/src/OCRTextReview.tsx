import { useEffect, useState } from 'react';
import PlateCrop, { useLoadedImage } from './PlateCrop';

type Annotation = {
  id: number; bbox_xyxy_pixels: number[]; ai_detection: boolean;
  ai_result?: string | null; verified_result?: string | null; detection_confidence?: number | null;
};
type ImageRecord = {
  id: string; filename: string; width: number; height: number; file_url: string;
  plate_count: number; verified_text_count: number; detection_status: string; ocr_status: string;
  detection_ready: boolean; ocr_ready: boolean;
  annotations?: Annotation[];
};

const FILTERS = [
  { value: 'ocr_missing', label: 'Needs transcription (boxes verified, no text)' },
  { value: 'ocr_partial', label: 'Partly transcribed' },
  { value: 'ocr_verified', label: 'Transcription verified' },
  { value: 'ocr_blocked', label: 'Blocked — boxes not verified yet' },
  { value: 'detection_verified', label: 'All detection-verified samples' },
  { value: 'all', label: 'All samples' },
];

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

/**
 * OCR ground-truth editor.
 *
 * Only the transcription is written here; boxes and the image review status are
 * left alone, so correcting text never discards detection verification.
 */
export default function OCRTextReview({ source, onOpenDetection }: { source: string; onOpenDetection: (imageId: string) => void }) {
  const [filter, setFilter] = useState('ocr_missing');
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [current, setCurrent] = useState<ImageRecord | null>(null);
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const image = useLoadedImage(current?.file_url);

  const load = (nextFilter: string, preferredId?: string) => {
    setError('');
    request<{ images: ImageRecord[] }>(`/api/images?source=${source}&readiness=${nextFilter}&limit=200`)
      .then(async (data) => {
        setImages(data.images);
        const target = data.images.find((item) => item.id === preferredId) || data.images[0] || null;
        if (target) {
          const detail = await request<ImageRecord>(`/api/images/${target.id}`);
          setCurrent(detail);
        } else setCurrent(null);
      })
      .catch((e: Error) => setError(e.message));
  };

  useEffect(() => { load(filter); }, [filter, source]);

  useEffect(() => {
    const next: Record<number, string> = {};
    (current?.annotations || []).forEach((item) => { next[item.id] = item.verified_result || ''; });
    setDrafts(next);
  }, [current?.id, current?.annotations]);

  const move = (direction: number) => {
    if (!current) return;
    const index = images.findIndex((item) => item.id === current.id);
    const next = images[index + direction];
    if (!next) return;
    request<ImageRecord>(`/api/images/${next.id}`).then(setCurrent).catch((e: Error) => setError(e.message));
  };

  const save = async () => {
    if (!current?.annotations?.length) return;
    setBusy(true); setError(''); setMessage('');
    try {
      const plates = current.annotations.map((item) => ({
        annotation_id: item.id,
        verified_result: drafts[item.id] ?? item.verified_result ?? '',
      }));
      const data = await request<{ image: ImageRecord }>(`/api/images/${current.id}/ocr-text`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plates }),
      });
      setCurrent(data.image);
      const saved = data.image.verified_text_count ?? 0;
      setMessage(`Saved ground-truth text for ${current.filename} · ${saved} of ${current.plate_count} plates now have verified text. Detection verification was not changed.`);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  return <section className="ocr-text-review">
    <header className="samples-head">
      <div>
        <h3>OCR text</h3>
        <p>Type the exact characters on each plate. A model prediction is shown for reference only and is never stored as ground truth until you save it here.</p>
      </div>
      <label className="review-filter">Show
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          {FILTERS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      </label>
    </header>

    {error && <p className="recognition-error" role="alert">{error}</p>}
    {message && <p className="recognition-message" role="status">{message}</p>}

    <div className="ocr-text-body">
      <aside className="ocr-text-list">
        {!images.length && !error && <p className="samples-empty">No samples match this filter.</p>}
        {images.map((item) => <button type="button" key={item.id}
          className={`ocr-text-item ${current?.id === item.id ? 'is-current' : ''}`}
          onClick={() => request<ImageRecord>(`/api/images/${item.id}`).then(setCurrent).catch((e: Error) => setError(e.message))}>
          <strong>{item.filename}</strong>
          <span>{item.verified_text_count}/{item.plate_count} plates with text · {item.ocr_status.replace('_', ' ')}</span>
        </button>)}
      </aside>

      <div className="ocr-text-detail">
        {!current && <p className="samples-empty">Select a sample on the left.</p>}
        {current && <>
          <div className="ocr-text-detail-head">
            <strong>{current.filename}</strong>
            <span className={`pill pill-${current.detection_ready ? 'ok' : 'warn'}`}>detection: {current.detection_status.replace('_', ' ')}</span>
            <span className={`pill pill-${current.ocr_ready ? 'ok' : 'warn'}`}>ocr: {current.ocr_status.replace('_', ' ')}</span>
            <span className="spacer" />
            <button type="button" onClick={() => move(-1)} aria-label="Previous sample">←</button>
            <button type="button" onClick={() => move(1)} aria-label="Next sample">→</button>
            <button type="button" onClick={() => onOpenDetection(current.id)}>View boxes</button>
          </div>

          {!current.annotations?.length && <p className="samples-empty">This image has no plate boxes yet. Draw them in the Detection view first.</p>}

          <ul className="ocr-plate-list">
            {(current.annotations || []).map((plate, index) => <li key={plate.id} className="ocr-plate">
              <div className="ocr-plate-crop">
                <PlateCrop image={image} box={plate.bbox_xyxy_pixels} />
                <span className="ocr-plate-index">Plate {index + 1}</span>
              </div>
              <div className="ocr-plate-fields">
                <label>Model prediction
                  <input value={plate.ai_result || ''} readOnly placeholder="No prediction" className="ocr-prediction" />
                </label>
                <label>Verified ground truth
                  <input
                    value={drafts[plate.id] ?? ''}
                    onChange={(e) => setDrafts((old) => ({ ...old, [plate.id]: e.target.value.toUpperCase() }))}
                    placeholder="e.g. 51A-1234"
                    maxLength={32}
                    aria-label={`Verified text for plate ${index + 1}`}
                  />
                </label>
                <span className="ocr-plate-meta">
                  box [{plate.bbox_xyxy_pixels.map((v) => Math.round(v)).join(', ')}]
                  {plate.detection_confidence != null && ` · detector ${(plate.detection_confidence * 100).toFixed(1)}%`}
                  {plate.ai_result && plate.verified_result && plate.ai_result !== plate.verified_result && ' · model differs from your text'}
                </span>
              </div>
            </li>)}
          </ul>

          <div className="ocr-text-actions">
            <button className="primary" onClick={() => void save()} disabled={busy || !current.annotations?.length}>{busy ? 'Saving…' : 'Save text'}</button>
            <span className="hint">Saving writes only the text above. Box verification and the image review status are untouched.</span>
          </div>
        </>}
      </div>
    </div>
  </section>;
}