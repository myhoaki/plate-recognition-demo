import { useEffect, useRef, useState } from 'react';

/**
 * Draw the region of `image` inside `box` onto a canvas.
 *
 * Shared by the Detection view and the OCR text view so a crop is produced the
 * same way everywhere: a plain pixel slice of the original image, clipped to the
 * image bounds. No model is involved.
 */
export default function PlateCrop({ image, box, className = 'plate-crop' }: { image: HTMLImageElement | null; box: number[]; className?: string }) {
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
  return <canvas className={className} ref={canvasRef} aria-label="Cropped plate image" />;
}

/** Load an image URL into an HTMLImageElement, exposing it once decoded. */
export function useLoadedImage(url: string | null | undefined) {
  const [element, setElement] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!url) { setElement(null); return; }
    let cancelled = false;
    const image = new Image();
    image.onload = () => { if (!cancelled) setElement(image); };
    image.src = url;
    return () => { cancelled = true; };
  }, [url]);
  return element;
}