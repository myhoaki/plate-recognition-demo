import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from 'react';
import { Circle, Group, Image as KImage, Layer, Rect, Stage, Text } from 'react-konva';
import type Konva from 'konva';

export type CanvasBox = {
  bbox_xyxy_pixels: number[];
  verified_result: string;
  ai_detection: boolean;
  ai_result?: string | null;
  detection_confidence?: number | null;
  ocr_confidence?: number | null;
  country?: string | null;
  country_confidence?: number | null;
};

type Props = {
  imageEl: HTMLImageElement | null;
  image: { id: string; width: number; height: number } | null;
  boxes: CanvasBox[];
  selectedIndex: number | null;
  /** draw: drag empty space to add a box. select: only pick and move existing boxes. */
  mode: 'select' | 'draw';
  minBoxSize?: number;
  disabled?: boolean;
  onSelect: (index: number | null) => void;
  /** Snapshot the pre-drag state so undo returns to where the drag began. */
  onEditStart: () => void;
  /** Live update while dragging; does not create an undo entry. */
  onChange: (boxes: CanvasBox[]) => void;
  /** Completed edit; creates one undo entry. */
  onCommit: (boxes: CanvasBox[]) => void;
  /** Reports the effective zoom so the toolbar can show it. */
  onZoomChange?: (zoom: number) => void;
};

const MIN_ZOOM = 0.2;
const MAX_ZOOM = 20;
const MIN_BOX_PX = 4;
// Pointer travel (viewport px) before an empty-space press becomes a drag.
const DRAG_THRESHOLD = 3;
const FILL = 'rgba(2, 6, 23, 0.55)';

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const round = (value: number) => Math.round(value * 100) / 100;

function normalizeBox(start: [number, number], end: [number, number], width: number, height: number, minSize: number) {
  const x1 = clamp(Math.min(start[0], end[0]), 0, width);
  const y1 = clamp(Math.min(start[1], end[1]), 0, height);
  const x2 = clamp(Math.max(start[0], end[0]), 0, width);
  const y2 = clamp(Math.max(start[1], end[1]), 0, height);
  // Never produce a zero-area or inverted box; fall back to a minimum size that
  // still stays inside the image.
  const w = Math.max(x2 - x1, minSize);
  const h = Math.max(y2 - y1, minSize);
  return [round(clamp(x1, 0, Math.max(0, width - w))), round(clamp(y1, 0, Math.max(0, height - h))),
    round(clamp(x1 + w, 0, width)), round(clamp(y1 + h, 0, height))];
}

/**
 * Annotation canvas with a viewport transform (zoom + pan) applied to a single
 * Konva layer. Shapes therefore use raw image-pixel coordinates and Konva owns
 * the screen mapping, which keeps boxes aligned through every transform.
 */
export type AnnotationCanvasHandle = {
  /** Fit the whole image into the viewport and reset zoom to 100% of fit. */
  fit: () => void;
  /** Multiply the current zoom, keeping the viewport centre fixed. */
  zoomBy: (factor: number) => void;
};

const AnnotationCanvas = forwardRef<AnnotationCanvasHandle, Props>(function AnnotationCanvas(
  { imageEl, image, boxes, selectedIndex, mode, minBoxSize = MIN_BOX_PX, disabled, onSelect, onEditStart, onChange, onCommit, onZoomChange }, ref) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const layerRef = useRef<Konva.Layer>(null);
  const [viewport, setViewport] = useState({ width: 800, height: 520 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [spaceDown, setSpaceDown] = useState(false);
  const [panning, setPanning] = useState(false);
  const [draft, setDraft] = useState<[number, number, number, number] | null>(null);
  const dragOrigin = useRef<[number, number] | null>(null);
  const movedWhileDown = useRef(false);
  // Manual pan state. Konva's own stage dragging is deliberately not used: its
  // `draggable` flag is read when the drag starts, so setting it from within a
  // mousedown handler is always one frame too late for middle-button drags.
  const panDrag = useRef<{ sx: number; sy: number; px: number; py: number } | null>(null);
  // A press on empty space is ambiguous until the pointer moves: it may become an
  // image drag (pan), a new box (Draw mode), or a plain click that deselects.
  // The gesture stays "unresolved" until it exceeds DRAG_THRESHOLD.
  const gesture = useRef<{
    sx: number; sy: number; px: number; py: number;
    resolved: null | 'pan' | 'draw';
  } | null>(null);
  // Box move / handle resize are driven manually rather than by Konva's
  // `draggable`. Konva drags write node.x()/y() directly while React re-renders
  // those same props from state; the two fight and a handle drag collapsed the
  // box to its minimum size. Tracking the original geometry in a ref makes every
  // drag absolute instead of cumulative, so it cannot drift or collapse.
  const boxDrag = useRef<{
    kind: 'move' | 'resize';
    index: number;
    corner: number;
    origin: [number, number];
    box: number[];
  } | null>(null);
  const dragOutput = useRef<CanvasBox[] | null>(null);

  // Fit scale: the multiplier applied to image pixels to fill the viewport.
  const fitScale = image
    ? Math.min(viewport.width / image.width, viewport.height / image.height)
    : 1;
  const viewScale = fitScale * zoom;

  useLayoutEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const measure = () => {
      // clientWidth/Height is the CONTENT box. getBoundingClientRect() includes
      // the 1px border, so using it here made the canvas 2px wider/taller than
      // the wrapper every cycle: ResizeObserver then fired again and the viewer
      // grew without bound until the page scrolled and boxes left the screen.
      const width = element.clientWidth;
      const height = element.clientHeight;
      if (width <= 0 || height <= 0) return;
      setViewport((old) => (old.width === width && old.height === height ? old : { width, height }));
    };
    measure();
    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure);
      return () => window.removeEventListener('resize', measure);
    }
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // Reset the view for each new image.
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setDraft(null);
  }, [image?.id]);

  const centre = useCallback(() => {
    if (!image) return;
    const fit = Math.min(viewport.width / image.width, viewport.height / image.height);
    setZoom(1);
    setPan({ x: (viewport.width - image.width * fit) / 2, y: (viewport.height - image.height * fit) / 2 });
  }, [image, viewport]);

  // Space is a momentary pan modifier, the convention in annotation tools.
  useEffect(() => {
    const isTyping = () => {
      const el = document.activeElement as HTMLElement | null;
      return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
    };
    const down = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !isTyping()) {
        event.preventDefault();
        setSpaceDown(true);
      }
    };
    const up = (event: KeyboardEvent) => { if (event.code === 'Space') setSpaceDown(false); };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); };
  }, []);

  const clampPan = useCallback((next: { x: number; y: number }, scale: number) => {
    if (!image) return next;
    // Keep at least a sliver of the image reachable so the user cannot get lost.
    const contentW = image.width * scale;
    const contentH = image.height * scale;
    const margin = 80;
    return {
      x: clamp(next.x, viewport.width - contentW - margin, margin),
      y: clamp(next.y, viewport.height - contentH - margin, margin),
    };
  }, [image, viewport]);

  const zoomBy = useCallback((factor: number, anchor?: { x: number; y: number }) => {
    const stage = stageRef.current;
    if (!stage || !image) return;
    const pointer = anchor || stage.getPointerPosition();
    if (!pointer) return;
    const stageX = stage.x();
    const stageY = stage.y();
    const worldX = (pointer.x - stageX) / viewScale;
    const worldY = (pointer.y - stageY) / viewScale;
    const nextZoom = clamp(zoom * factor, MIN_ZOOM, MAX_ZOOM);
    if (nextZoom === zoom) return;
    const nextScale = fitScale * nextZoom;
    setZoom(nextZoom);
    // Keep the point under the cursor fixed while the scale changes.
    setPan(clampPan({ x: pointer.x - worldX * nextScale, y: pointer.y - worldY * nextScale }, nextScale));
  }, [clampPan, fitScale, image, viewScale, zoom]);

  // Wheel zoom must not scroll the page; React's handler is passive, so bind natively.
  useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = element.getBoundingClientRect();
      zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12, { x: event.clientX - rect.left, y: event.clientY - rect.top });
    };
    element.addEventListener('wheel', onWheel, { passive: false });
    return () => element.removeEventListener('wheel', onWheel);
  }, [zoomBy]);

  useImperativeHandle(ref, () => ({
    fit: () => centre(),
    zoomBy: (factor: number) => {
      const stage = stageRef.current;
      if (!stage) return;
      zoomBy(factor, { x: viewport.width / 2, y: viewport.height / 2 });
    },
  }), [centre, zoomBy, viewport.width, viewport.height]);

  const panActive = panning || spaceDown;
  // Only offer image dragging when the image is actually larger than the
  // viewport; at fit there is nothing useful to navigate to.
  const pannable = !!image && (image.width * viewScale > viewport.width + 1 || image.height * viewScale > viewport.height + 1);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    stage.position({ x: pan.x, y: pan.y });
    stage.scale({ x: viewScale, y: viewScale });
    layerRef.current?.batchDraw();
  }, [pan, viewScale]);

  useEffect(() => {
    onZoomChange?.(Math.round(zoom * 100));
  }, [zoom, onZoomChange]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    stage.position({ x: pan.x, y: pan.y });
    stage.scale({ x: viewScale, y: viewScale });
    stage.batchDraw();
  }, [pan, viewScale]);

  const pointerInImage = useCallback((): [number, number] | null => {
    const layer = layerRef.current;
    if (!layer) return null;
    const point = layer.getRelativePointerPosition();
    return point ? [point.x, point.y] : null;
  }, []);

  const handleMouseDown = (event: Konva.KonvaEventObject<MouseEvent>) => {
    if (!image || disabled) return;
    // Stop the browser's native image drag and text selection for every gesture.
    event.evt.preventDefault();
    const middle = event.evt.button === 1;
    if (middle || spaceDown) {
      const pointer = stageRef.current?.getPointerPosition();
      if (pointer) panDrag.current = { sx: pointer.x, sy: pointer.y, px: pan.x, py: pan.y };
      setPanning(true);
      return;
    }
    if (event.evt.button !== 0) return;
    // Highest precedence: a press on a box or a resize handle belongs to that
    // box, so image dragging never starts here and box editing is untouched.
    if (event.target !== event.target.getStage()) return;
    movedWhileDown.current = false;
    const pointer = stageRef.current?.getPointerPosition();
    // Empty space: remember where the press happened and decide on first move.
    gesture.current = pointer ? { sx: pointer.x, sy: pointer.y, px: pan.x, py: pan.y, resolved: null } : null;
  };

  const handleMouseMove = () => {
    // Resolve an ambiguous empty-space press once it travels far enough.
    const pending = gesture.current;
    if (pending && !pending.resolved) {
      const pointer = stageRef.current?.getPointerPosition();
      if (pointer) {
        const travelX = pointer.x - pending.sx;
        const travelY = pointer.y - pending.sy;
        if (Math.abs(travelX) > DRAG_THRESHOLD || Math.abs(travelY) > DRAG_THRESHOLD) {
          movedWhileDown.current = true;
          // Draw mode only draws while the image fits the viewport; once the
          // image overflows, dragging empty space moves the image instead.
          if (mode === 'draw' && !pannable) {
            pending.resolved = 'draw';
            const start = pointerInImage();
            if (start) { dragOrigin.current = start; setDraft([start[0], start[1], start[0], start[1]]); }
          } else {
            pending.resolved = 'pan';
            panDrag.current = { sx: pending.sx, sy: pending.sy, px: pending.px, py: pending.py };
            setPanning(true);
          }
        }
      }
    }
    // Panning takes priority, so a pan gesture can never create or move a box.
    const panStart = panDrag.current;
    if (panStart) {
      const pointer = stageRef.current?.getPointerPosition();
      if (!pointer) return;
      setPan(clampPan({
        x: panStart.px + (pointer.x - panStart.sx),
        y: panStart.py + (pointer.y - panStart.sy),
      }, viewScale));
      return;
    }
    const active = boxDrag.current;
    if (active && image) {
      const point = pointerInImage();
      if (!point) return;
      movedWhileDown.current = true;
      let next: CanvasBox[];
      if (active.kind === 'move') {
        const [bx1, by1, bx2, by2] = active.box;
        const width = bx2 - bx1;
        const height = by2 - by1;
        // Absolute: measured from the geometry at drag start, never accumulated.
        const left = clamp(active.box[0] + (point[0] - active.origin[0]), 0, Math.max(0, image.width - width));
        const top = clamp(active.box[1] + (point[1] - active.origin[1]), 0, Math.max(0, image.height - height));
        const coords = [round(left), round(top), round(left + width), round(top + height)];
        next = boxes.map((box, i) => i === active.index ? { ...box, bbox_xyxy_pixels: coords } : box);
      } else {
        // The fixed anchor is the corner OPPOSITE the one being dragged
        // (0=TL, 1=TR, 2=BL, 3=BR). Corners 0/1 move the left edge, so the right
        // edge is anchored; corners 2/3 move the right edge, so the left is.
        const anchor: [number, number] = [
          active.corner === 1 || active.corner === 3 ? active.box[0] : active.box[2],
          active.corner === 2 || active.corner === 3 ? active.box[1] : active.box[3],
        ];
        const coords = normalizeBox(anchor, point, image.width, image.height, minBoxSize);
        next = boxes.map((box, i) => i === active.index ? { ...box, bbox_xyxy_pixels: coords } : box);
      }
      dragOutput.current = next;
      onChange(next);
      return;
    }
    if (dragOrigin.current && draft) {
      const point = pointerInImage();
      if (!point) return;
      movedWhileDown.current = true;
      setDraft([dragOrigin.current[0], dragOrigin.current[1], point[0], point[1]]);
    }
  };

  /** Begin a manual move/resize gesture from a pointer position in image space. */
  const startBoxDrag = (kind: 'move' | 'resize', index: number, corner: number) => {
    const point = pointerInImage();
    if (!point || !boxes[index]) return;
    onSelect(index);
    onEditStart();
    boxDrag.current = { kind, index, corner, origin: point, box: [...boxes[index].bbox_xyxy_pixels] };
    dragOutput.current = null;
    movedWhileDown.current = false;
  };

  const handleMouseUp = () => {
    const pending = gesture.current;
    gesture.current = null;
    panDrag.current = null;
    setPanning(false);
    const active = boxDrag.current;
    if (active) {
      boxDrag.current = null;
      if (dragOutput.current && movedWhileDown.current) onCommit(dragOutput.current);
      dragOutput.current = null;
      return;
    }
    if (dragOrigin.current && draft) {
      const point = pointerInImage() || [draft[2], draft[3]];
      const [x1, y1, x2, y2] = normalizeBox(dragOrigin.current, point, image?.width ?? 0, image?.height ?? 0, minBoxSize);
      // A click (not a drag) must not create a box.
      if (movedWhileDown.current && (x2 - x1 >= minBoxSize) && (y2 - y1 >= minBoxSize)) {
        const next = [...boxes, { bbox_xyxy_pixels: [x1, y1, x2, y2], verified_result: '', ai_detection: false }];
        onCommit(next);
        onSelect(next.length - 1);
      }
    }
    dragOrigin.current = null;
    setDraft(null);
    // An unresolved gesture was a plain click on empty space: keep the previous
    // behaviour of clearing the selection.
    if (pending && !pending.resolved) onSelect(null);
    movedWhileDown.current = false;
  };

  // Bring the selected box into view when selection comes from the list.
  useEffect(() => {
    if (selectedIndex == null || !image || !boxes[selectedIndex]) return;
    const [x1, y1, x2, y2] = boxes[selectedIndex].bbox_xyxy_pixels;
    const cx = (x1 + x2) / 2 * viewScale;
    const cy = (y1 + y2) / 2 * viewScale;
    if (cx + pan.x >= 0 && cx + pan.x <= viewport.width && cy + pan.y >= 0 && cy + pan.y <= viewport.height) return;
    setPan({ x: viewport.width / 2 - cx, y: viewport.height / 2 - cy });
  }, [selectedIndex]);

  // Release outside the canvas must not leave a gesture stuck. Konva only sees
  // events over the stage, so mirror pointerup/pointercancel at the window.
  const endGestureRef = useRef<() => void>(() => undefined);
  endGestureRef.current = handleMouseUp;
  useEffect(() => {
    const end = () => {
      if (panDrag.current || boxDrag.current || dragOrigin.current || gesture.current) endGestureRef.current();
    };
    window.addEventListener('pointerup', end);
    window.addEventListener('pointercancel', end);
    return () => {
      window.removeEventListener('pointerup', end);
      window.removeEventListener('pointercancel', end);
    };
  }, []);

  const inverse = viewScale ? 1 / viewScale : 1;

  return <div ref={wrapRef} className={`canvas-wrap ${panActive ? 'is-panning' : ''} mode-${mode} ${pannable ? 'is-pannable' : ''}`}
    onContextMenu={(event) => event.preventDefault()}>
    <Stage ref={stageRef} width={viewport.width} height={viewport.height}
      onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchStart={handleMouseDown as never} onTouchMove={handleMouseMove as never} onTouchEnd={handleMouseUp as never}>
      {/* The layer MUST stay listening: Konva skips a non-listening layer during
          hit detection, which would make every box and handle unclickable. The
          image shape alone is non-listening, so empty-space clicks fall through
          to the stage. */}
      <Layer ref={layerRef}>
        {imageEl && image && <>
          {/* listening={false} so clicks on empty pixels reach the stage and can
              start a draw gesture; boxes are drawn after, so they hit-test first. */}
          <KImage image={imageEl} width={image.width} height={image.height} listening={false} />
          {boxes.map((box, index) => {
            const [x1, y1, x2, y2] = box.bbox_xyxy_pixels;
            const selected = index === selectedIndex;
            const label = box.ai_result || box.verified_result || `Plate ${index + 1}`;
            return <Group key={index} listening>
              <Rect x={x1} y={y1} width={x2 - x1} height={y2 - y1}
                stroke={selected ? '#22d3ee' : box.ai_detection ? '#f59e0b' : '#22c55e'}
                strokeWidth={(selected ? 3 : 2) * inverse}
                dash={box.ai_detection ? [6 * inverse, 4 * inverse] : undefined}
                fill={selected ? 'rgba(34,211,238,0.10)' : 'transparent'}
                hitStrokeWidth={14 * inverse}
                onMouseDown={(event) => { event.cancelBubble = true; if (!disabled) startBoxDrag('move', index, 0); }}
                onTouchStart={(event) => { event.cancelBubble = true; if (!disabled) startBoxDrag('move', index, 0); }}
              />
              {selected && <Rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill={FILL} listening={false} />}
              <Text x={x1} y={Math.max(0, y1 - 22 * inverse)} width={Math.max(80 * inverse, x2 - x1)} align="left"
                text={label} fontSize={13 * inverse} fill="#f8fafc" listening={false}
                padding={3 * inverse} />
              {selected && [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].map(([hx, hy], corner) =>
                <Circle key={corner} x={hx} y={hy} radius={6 * inverse} fill="#22d3ee" stroke="#0e7490" strokeWidth={1.5 * inverse}
                  hitStrokeWidth={10 * inverse}
                  onMouseDown={(event) => { event.cancelBubble = true; if (!disabled) startBoxDrag('resize', index, corner); }}
                  onTouchStart={(event) => { event.cancelBubble = true; if (!disabled) startBoxDrag('resize', index, corner); }}
                />)}
            </Group>;
          })}
          {draft && (() => {
            const [x1, y1, x2, y2] = normalizeBox([draft[0], draft[1]], [draft[2], draft[3]], image.width, image.height, minBoxSize);
            return <Rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} stroke="#22d3ee" strokeWidth={2 * inverse}
              dash={[6 * inverse, 4 * inverse]} listening={false} />;
          })()}
        </>}
      </Layer>
    </Stage>
  </div>;
});

export default AnnotationCanvas;