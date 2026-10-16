import { useCallback, useEffect, useState } from 'react';

type Sample = { id: string; filename: string; file_url: string; plate_text: string; status: 'pending' | 'verified' | 'rejected'; is_crop: boolean; available: boolean; error?: string | null };
async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Request failed (${response.status})`); }
  return response.json();
}

export default function OCRDatasetReview() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [index, setIndex] = useState(0);
  const [offset, setOffset] = useState(0);
  const [total, setTotal] = useState(0);
  const [text, setText] = useState('');
  const [margin, setMargin] = useState(0.08);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const current = samples[index];

  const load = useCallback(async (newOffset = offset) => {
    const data = await request<{ samples: Sample[]; total: number }>('/api/ocr-samples?limit=100&offset=' + newOffset);
    setSamples(data.samples); setTotal(data.total); setOffset(newOffset); setIndex(0);
  }, [offset]);
  useEffect(() => { void load(0).catch((error) => setMessage((error as Error).message)); }, []);
  useEffect(() => { setText(current?.plate_text || ''); }, [current?.id]);

  const go = (direction: number) => {
    if (current && text !== current.plate_text) { setMessage('Save the OCR label or choose a review status before navigating.'); return; }
    if (direction > 0 && index === samples.length - 1 && offset + samples.length < total) void load(offset + 100);
    else if (direction < 0 && index === 0 && offset > 0) void load(Math.max(0, offset - 100)).then(() => setIndex(99));
    else setIndex((old) => Math.max(0, Math.min(samples.length - 1, old + direction)));
  };
  const save = async (status: Sample['status']) => {
    if (!current) return;
    setBusy(true);
    try {
      await request(`/api/ocr-samples/${current.id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ plate_text: text, status }) });
      setSamples((all) => all.map((sample) => sample.id === current.id ? { ...sample, plate_text: text, status } : sample));
      setMessage(status === 'verified' ? 'OCR ground truth verified.' : status === 'rejected' ? 'Sample flagged as unreadable or rejected.' : 'Sample left pending.');
    } catch (error) { setMessage((error as Error).message); }
    finally { setBusy(false); }
  };

  return <section className="ocr-review-panel"><header><div><h3>OCR dataset review</h3><p>Check imported crop/text rows from raw-dataset-ocr/annotations.csv. Keep ambiguous text pending or reject it; verification preserves the transcription exactly.</p></div><button onClick={() => void load(0)} disabled={busy}>Reload CSV</button></header>
    <div className="ocr-review-toolbar"><button onClick={() => go(-1)} disabled={!total || (!offset && index === 0)}>← Previous</button><span>{total ? `${offset + index + 1} / ${total}` : 'No OCR rows'}</span><button onClick={() => go(1)} disabled={!total || offset + index >= total - 1}>Next →</button><label>Crop margin<input type="range" min="0" max="0.3" step="0.02" value={margin} onChange={(event) => setMargin(Number(event.target.value))} /><small>{Math.round(margin * 100)}%</small></label></div>
    {current ? <div className="ocr-review-body"><div><strong>{current.filename}</strong>{current.available ? <img src={`${current.file_url}?margin=${margin}&v=${current.id}`} alt={`OCR sample ${current.filename}`} /> : <p className="training-error">{current.error || 'Image unavailable'}</p>}</div><div className="ocr-review-form"><label>Ground-truth plate text<input value={text} onChange={(event) => setText(event.target.value)} maxLength={32} autoComplete="off" /></label><p>Status: <b>{current.status}</b> · {current.is_crop ? 'Standalone crop' : 'Crop from source image'}</p><div><button className="primary" onClick={() => void save('verified')} disabled={busy || !current.available || !text}>Verify text</button><button onClick={() => void save('pending')} disabled={busy}>Keep pending</button><button onClick={() => void save('rejected')} disabled={busy}>Reject / unreadable</button></div></div></div> : <p>No OCR samples found. Add rows to raw-dataset-ocr/annotations.csv.</p>}
    {message && <p className="recognition-message" role="status">{message}</p>}
  </section>;
}
