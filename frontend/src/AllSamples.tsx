import { useEffect, useState } from 'react';

type ImageRecord = {
  id: string; filename: string; plate_count: number; verified_text_count: number;
  detection_status: string; ocr_status: string; detection_ready: boolean; ocr_ready: boolean; ready_for_both: boolean;
};

type Summary = {
  total_images: number; with_boxes: number;
  detection_verified: number; detection_needs_review: number; detection_excluded: number;
  ocr_verified: number; ocr_partial: number; ocr_missing: number; ocr_blocked: number;
  ready_for_both: number; plates_total: number; plates_verified_text: number; plates_prediction_only: number;
};

const READINESS = [
  { value: 'all', label: 'All samples' },
  { value: 'detection_needs_review', label: 'Detection needs review' },
  { value: 'detection_verified', label: 'Detection verified' },
  { value: 'ocr_missing', label: 'OCR text missing (boxes verified, no text yet)' },
  { value: 'ocr_partial', label: 'OCR partially transcribed' },
  { value: 'ocr_blocked', label: 'OCR blocked (boxes not verified)' },
  { value: 'ocr_verified', label: 'OCR verified' },
  { value: 'ready_for_both', label: 'Ready for both tasks' },
];

const STATUS_LABEL: Record<string, string> = {
  verified: 'Verified', needs_review: 'Needs review', blocked: 'Boxes not verified',
  missing: 'No text yet', partial: 'Partly transcribed', no_boxes: 'No boxes', rejected: 'Rejected', no_plate: 'No plate',
};

async function request<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

/** Overview of detection vs OCR readiness, with counts taken from the backend. */
export default function AllSamples({ source, onOpen }: { source: string; onOpen: (imageId: string) => void }) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [readiness, setReadiness] = useState('all');
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    request<Summary>(`/api/verification/summary?source=${source}`)
      .then(setSummary).catch((e: Error) => setError(e.message));
  }, [source]);

  useEffect(() => {
    request<{ images: ImageRecord[] }>(`/api/images?source=${source}&readiness=${readiness}&limit=500`)
      .then((data) => setImages(data.images)).catch((e: Error) => setError(e.message));
  }, [source, readiness]);

  if (error) return <p className="recognition-error" role="alert">{error}</p>;

  const tiles: [string, number, string][] = summary ? [
    ['Images', summary.total_images, ''],
    ['Detection verified', summary.detection_verified, 'ok'],
    ['Detection needs review', summary.detection_needs_review, 'warn'],
    ['OCR verified', summary.ocr_verified, 'ok'],
    ['OCR missing text', summary.ocr_missing, 'warn'],
    ['OCR partial', summary.ocr_partial, 'warn'],
    ['Ready for both', summary.ready_for_both, 'ok'],
    ['Plates with text', summary.plates_verified_text, `of ${summary.plates_total}`],
  ] : [];

  return <section className="all-samples">
    <header className="samples-head">
      <div>
        <h3>All samples</h3>
        <p>Detection and OCR readiness are counted independently. A sample can be detection-ready while its text is still missing.</p>
      </div>
      <label className="review-filter">Show
        <select value={readiness} onChange={(e) => setReadiness(e.target.value)}>
          {READINESS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      </label>
    </header>

    {summary && <div className="readiness-tiles">
      {tiles.map(([label, value, tone]) => <div key={label} className={`readiness-tile tone-${tone || 'plain'}`}>
        <span>{label}</span><strong>{value.toLocaleString()}{tone.startsWith('of ') ? '' : ''}</strong>
        {tone.startsWith('of ') && <small>{tone.slice(3)}</small>}
      </div>)}
    </div>}

    {summary && summary.plates_prediction_only > 0 && <p className="samples-note">
      {summary.plates_prediction_only.toLocaleString()} plate{summary.plates_prediction_only === 1 ? '' : 's'} still carry only a model prediction.
      Predictions are never treated as ground truth until you verify them in the OCR Text view.
    </p>}

    <div className="samples-table-wrap">
      <table className="samples-table">
        <thead><tr><th>File</th><th>Plates</th><th>Detection</th><th>OCR text</th><th></th></tr></thead>
        <tbody>
          {!images.length && <tr><td colSpan={5} className="samples-empty">No samples match this filter.</td></tr>}
          {images.map((image) => <tr key={image.id}>
            <td className="samples-file">{image.filename}</td>
            <td>{image.plate_count}<small> · {image.verified_text_count} with text</small></td>
            <td><span className={`pill pill-${image.detection_ready ? 'ok' : image.detection_status === 'needs_review' ? 'warn' : 'muted'}`}>{STATUS_LABEL[image.detection_status] ?? image.detection_status}</span></td>
            <td><span className={`pill pill-${image.ocr_ready ? 'ok' : image.ocr_status === 'blocked' ? 'muted' : 'warn'}`}>{STATUS_LABEL[image.ocr_status] ?? image.ocr_status}</span></td>
            <td><button type="button" onClick={() => onOpen(image.id)}>Open</button></td>
          </tr>)}
        </tbody>
      </table>
      {images.length >= 500 && <p className="samples-note">Showing the first 500 matches. Narrow the filter to see the rest.</p>}
    </div>
  </section>;
}