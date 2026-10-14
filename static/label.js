(() => {
  const fileInput = document.getElementById('label-image');
  const rawSelect = document.getElementById('label-rawimage');
  const uploadPanel = document.getElementById('label-upload-panel');
  const rawPanel = document.getElementById('label-raw-panel');
  const canvas = document.getElementById('label-canvas');
  const context = canvas.getContext('2d');
  const canvasWrap = document.getElementById('canvas-wrap');
  const rawImageNav = document.getElementById('raw-image-nav');
  const objectsSection = document.getElementById('objects-section');
  const objectList = document.getElementById('object-list');
  const saveButton = document.getElementById('save-labels');
  const detectButton = document.getElementById('detect-plates');
  const batchButton = document.getElementById('batch-detect');
  const batchStatus = document.getElementById('batch-status');
  const alertBox = document.getElementById('label-alert');
  const savedMessage = document.getElementById('saved-message');
  const image = new Image();
  let imageFile = null;
  let imageReady = false;
  let sourceType = 'upload';
  let selectedRawImage = '';
  let boxes = [];
  let drawing = null;
  let objectUrl = null;
  let detecting = false;
  let batchDetecting = false;
  let stopBatchRequested = false;
  let imageGeneration = 0;
  let userEditedImage = false;

  function showAlert(message, isError = true) {
    alertBox.textContent = message;
    alertBox.classList.toggle('is-error', isError);
    alertBox.hidden = false;
  }

  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * canvas.width / rect.width,
      y: (event.clientY - rect.top) * canvas.height / rect.height,
    };
  }

  function boxFromPixels(bounds) {
    const scaleX = canvas.width / image.naturalWidth;
    const scaleY = canvas.height / image.naturalHeight;
    return {
      x1: bounds[0] * scaleX,
      y1: bounds[1] * scaleY,
      x2: bounds[2] * scaleX,
      y2: bounds[3] * scaleY,
    };
  }

  function formatPercent(value) {
    if (value == null || value === '') return '—';
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${Math.round(numeric * 100)}%` : '—';
  }

  function addDetectionStats(box, detection) {
    box.country = detection.country || '';
    box.countryConfidence = detection.country_confidence ?? null;
    box.detectionConfidence = detection.detection_confidence ?? null;
    box.ocrConfidence = detection.ocr_confidence ?? null;
  }

  function boxOverlap(first, second) {
    const left = Math.max(first.x1, second.x1);
    const top = Math.max(first.y1, second.y1);
    const right = Math.min(first.x2, second.x2);
    const bottom = Math.min(first.y2, second.y2);
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
    const firstArea = Math.max(0, first.x2 - first.x1) * Math.max(0, first.y2 - first.y1);
    const secondArea = Math.max(0, second.x2 - second.x1) * Math.max(0, second.y2 - second.y1);
    return intersection / Math.max(1, firstArea + secondArea - intersection);
  }

  function draw() {
    if (!image.complete || !image.naturalWidth) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    boxes.forEach((box, index) => drawBox(box, index));
    if (drawing) drawBox({ ...drawing, text: '' }, boxes.length, true);
  }

  function drawBox(box, index, temporary = false) {
    const x = Math.min(box.x1, box.x2);
    const y = Math.min(box.y1, box.y2);
    const width = Math.abs(box.x2 - box.x1);
    const height = Math.abs(box.y2 - box.y1);
    context.save();
    context.strokeStyle = temporary ? '#ffffff' : '#d8f172';
    context.lineWidth = Math.max(2, canvas.width / 700);
    context.setLineDash(temporary ? [7, 5] : []);
    context.strokeRect(x, y, width, height);
    context.fillStyle = 'rgba(19, 35, 27, .78)';
    const title = `PLATE ${String(index + 1).padStart(2, '0')}`;
    context.font = `600 ${Math.max(13, canvas.width / 75)}px Arial`;
    const titleWidth = context.measureText(title).width + 14;
    context.fillRect(x, Math.max(0, y - 25), titleWidth, 23);
    context.fillStyle = '#ffffff';
    context.fillText(title, x + 7, Math.max(16, y - 9));
    context.restore();
  }

  function updateSaveState() {
    const allHaveText = boxes.length > 0 && boxes.every(box => box.text.trim() || (box.aiText || '').trim());
    saveButton.disabled = !imageReady || !allHaveText;
    detectButton.disabled = !imageReady || detecting || batchDetecting;
    if (batchButton) batchButton.disabled = detecting && !batchDetecting;
    document.getElementById('box-count').textContent = boxes.length;
  }

  function clearCurrentImage() {
    imageGeneration += 1;
    imageFile = null;
    imageReady = false;
    userEditedImage = false;
    boxes = [];
    canvasWrap.hidden = true;
    renderObjectList();
    draw();
  }

  function loadImage(source, filename, file = null) {
    const generation = ++imageGeneration;
    imageFile = file;
    imageReady = false;
    userEditedImage = false;
    boxes = [];
    savedMessage.hidden = true;
    alertBox.hidden = true;
    image.onload = () => {
      if (generation !== imageGeneration || userEditedImage) return;
      const scale = Math.min(1, 1600 / image.naturalWidth, 1200 / image.naturalHeight);
      canvas.width = Math.round(image.naturalWidth * scale);
      canvas.height = Math.round(image.naturalHeight * scale);
      canvasWrap.hidden = false;
      if (rawImageNav) rawImageNav.hidden = sourceType !== 'rawimages';
      document.getElementById('pick-title').textContent = filename;
      document.getElementById('pick-subtitle').textContent = `${image.naturalWidth} × ${image.naturalHeight} px`;
      imageReady = true;
      renderObjectList();
      draw();
      restoreSavedImageInfo(generation, filename);
    };
    image.onerror = () => {
      if (generation !== imageGeneration) return;
      imageReady = false;
      showAlert('This image could not be opened.');
      updateSaveState();
    };
    image.src = source;
    updateSaveState();
  }

  async function restoreSavedImageInfo(generation, filename) {
    try {
      const response = await fetch('/api/label-state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: sourceType, filename }),
      });
      if (!response.ok) return;
      const saved = await response.json();
      if (generation !== imageGeneration) return;

      const detections = Array.isArray(saved.detections) ? saved.detections : [];
      const groundTruth = Array.isArray(saved.ground_truth) ? saved.ground_truth : [];
      const savedBoxes = [];

      if (groundTruth.length) {
        const matchedDetections = new Set();
        groundTruth.forEach(item => {
          const box = boxFromPixels(item.bbox_xyxy_pixels);
          const matched = detections
            .map((detection, index) => ({ detection, index, overlap: boxOverlap(box, boxFromPixels(detection.bbox_xyxy_pixels)) }))
            .filter(candidate => !matchedDetections.has(candidate.index))
            .sort((first, second) => second.overlap - first.overlap)[0];
          const hasMatch = matched && matched.overlap >= 0.15;
          if (hasMatch) matchedDetections.add(matched.index);
          savedBoxes.push({
            ...box,
            text: item.ground_truth || '',
            aiText: hasMatch ? matched.detection.text || '' : '',
            aiDetection: false,
            country: hasMatch ? matched.detection.country || '' : item.country || '',
            countryConfidence: hasMatch ? matched.detection.country_confidence : item.country_confidence,
            detectionConfidence: hasMatch ? matched.detection.detection_confidence : item.detection_confidence,
            ocrConfidence: hasMatch ? matched.detection.ocr_confidence : item.ocr_confidence,
          });
        });
        detections.forEach((item, index) => {
          if (matchedDetections.has(index)) return;
          const savedBox = {
            ...boxFromPixels(item.bbox_xyxy_pixels),
            text: '',
            aiText: item.text || '',
            aiDetection: true,
          };
          addDetectionStats(savedBox, item);
          savedBoxes.push(savedBox);
        });
      } else if (detections.length) {
        detections.forEach(item => {
          const savedBox = {
            ...boxFromPixels(item.bbox_xyxy_pixels),
            text: '',
            aiText: item.text || '',
            aiDetection: true,
          };
          addDetectionStats(savedBox, item);
          savedBoxes.push(savedBox);
        });
      }

      if (savedBoxes.length) {
        // The user may draw while this request is in flight. Preserve those
        // boxes and merge in persisted detections that are not already there.
        const mergedBoxes = [...boxes];
        savedBoxes.forEach(savedBox => {
          const match = mergedBoxes
            .map(box => ({ box, overlap: boxOverlap(box, savedBox) }))
            .sort((first, second) => second.overlap - first.overlap)[0];
          if (match && match.overlap >= 0.15) {
            if (!match.box.aiText) match.box.aiText = savedBox.aiText;
            if (!match.box.text && savedBox.text && !userEditedImage) match.box.text = savedBox.text;
            if (!match.box.country) match.box.country = savedBox.country;
            if (match.box.countryConfidence == null) match.box.countryConfidence = savedBox.countryConfidence;
            if (match.box.detectionConfidence == null) match.box.detectionConfidence = savedBox.detectionConfidence;
            if (match.box.ocrConfidence == null) match.box.ocrConfidence = savedBox.ocrConfidence;
          } else {
            mergedBoxes.push(savedBox);
          }
        });
        boxes = mergedBoxes;
        renderObjectList();
        draw();
        const summary = groundTruth.length
          ? `Loaded ${groundTruth.length} saved ground-truth label${groundTruth.length === 1 ? '' : 's'}.`
          : `Loaded ${detections.length} saved AI suggestion${detections.length === 1 ? '' : 's'}.`;
        showAlert(`${summary} You can continue editing this image.`, false);
      }
    } catch (error) {
      console.error('Could not restore saved labels for this image.', error);
      if (generation === imageGeneration) {
        showAlert('Could not load this image’s saved boxes. Try changing images and returning to it.');
      }
    }
  }

  function renderObjectList() {
    objectList.replaceChildren();
    objectsSection.hidden = boxes.length === 0;
    boxes.forEach((box, index) => {
      const row = document.createElement('div');
      row.className = 'object-row';
      const number = document.createElement('span');
      number.className = 'object-number';
      number.textContent = String(index + 1).padStart(2, '0');
      const fields = document.createElement('div');
      fields.className = 'object-fields';
      if (box.aiText) {
        const suggestion = document.createElement('small');
        suggestion.className = 'ai-suggestion';
        suggestion.textContent = `AI guess: ${box.aiText}`;
        fields.append(suggestion);
      }
      if (box.aiDetection || box.detectionConfidence != null || box.country) {
        const stats = document.createElement('small');
        stats.className = 'detection-stats';
        const country = box.country
          ? `${box.country}${box.countryConfidence != null ? ` ${formatPercent(box.countryConfidence)}` : ''}`
          : 'Country unavailable';
        stats.textContent = `${country} · Detect ${formatPercent(box.detectionConfidence)} · OCR ${formatPercent(box.ocrConfidence)}`;
        fields.append(stats);
      }
      const input = document.createElement('input');
      input.type = 'text';
      input.maxLength = 32;
      input.autocomplete = 'off';
      input.placeholder = 'Enter verified plate text';
      input.setAttribute('aria-label', `Ground truth for plate ${index + 1}`);
      input.value = box.text;
      input.addEventListener('input', () => {
        box.text = input.value;
        userEditedImage = true;
        updateSaveState();
      });
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'remove-object';
      remove.setAttribute('aria-label', `Remove plate ${index + 1}`);
      remove.textContent = '×';
      remove.addEventListener('click', () => {
        userEditedImage = true;
        boxes.splice(index, 1);
        renderObjectList();
        draw();
        updateSaveState();
      });
      fields.append(input);
      row.append(number, fields, remove);
      objectList.append(row);
    });
    updateSaveState();
  }

  function loadUploadedFile(selectedFile) {
    if (!selectedFile) {
      clearCurrentImage();
      return;
    }
    if (!selectedFile.type.startsWith('image/')) {
      showAlert('Choose a supported image file.');
      fileInput.value = '';
      return;
    }
    if (selectedFile.size > 15 * 1024 * 1024) {
      showAlert('The image must be 15 MB or smaller.');
      fileInput.value = '';
      return;
    }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(selectedFile);
    loadImage(objectUrl, selectedFile.name, selectedFile);
  }

  function loadRawImage(filename) {
    if (!filename) {
      clearCurrentImage();
      return;
    }
    selectedRawImage = filename;
    loadImage(`/raw-images/${encodeURIComponent(filename)}`, filename);
  }

  function updateSourceMode() {
    sourceType = document.querySelector('input[name="label-source"]:checked').value;
    const useRaw = sourceType === 'rawimages';
    uploadPanel.hidden = useRaw;
    rawPanel.hidden = !useRaw;
    if (rawImageNav) rawImageNav.hidden = !useRaw;
    fileInput.disabled = useRaw;
    if (rawSelect) rawSelect.disabled = !useRaw;
    if (useRaw) {
      loadRawImage(rawSelect ? rawSelect.value : '');
    } else if (fileInput.files[0]) {
      loadUploadedFile(fileInput.files[0]);
    } else {
      selectedRawImage = '';
      clearCurrentImage();
    }
  }

  document.querySelectorAll('input[name="label-source"]').forEach(option => option.addEventListener('change', updateSourceMode));
  if (rawSelect) rawSelect.addEventListener('change', () => loadRawImage(rawSelect.value));
  fileInput.addEventListener('change', () => loadUploadedFile(fileInput.files[0]));

  detectButton.addEventListener('click', async () => {
    if (!imageReady || detecting) return;
    const generation = imageGeneration;
    const form = new FormData();
    form.append('source', sourceType);
    if (sourceType === 'upload') {
      if (!imageFile) return;
      form.append('image', imageFile, imageFile.name);
    } else {
      form.append('rawimage', selectedRawImage);
    }

    detecting = true;
    userEditedImage = true;
    detectButton.textContent = 'Detecting…';
    alertBox.hidden = true;
    updateSaveState();
    try {
      const response = await fetch('/api/label-detections', { method: 'POST', body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'AI detection failed.');
      if (generation !== imageGeneration) return;

      result.detections.forEach(detection => {
        const candidate = {
          ...boxFromPixels(detection.bbox_xyxy_pixels),
          text: '',
          aiText: detection.text || '',
          aiDetection: true,
        };
        addDetectionStats(candidate, detection);
        const bestMatch = boxes
          .map(box => ({ box, overlap: boxOverlap(box, candidate) }))
          .sort((first, second) => second.overlap - first.overlap)[0];
        if (bestMatch && bestMatch.overlap >= 0.15) {
          bestMatch.box.aiText = candidate.aiText;
          Object.assign(bestMatch.box, {
            country: candidate.country,
            countryConfidence: candidate.countryConfidence,
            detectionConfidence: candidate.detectionConfidence,
            ocrConfidence: candidate.ocrConfidence,
          });
        }
        else boxes.push(candidate);
      });
      renderObjectList();
      draw();
      if (result.detections.length) {
        showAlert(
          `AI suggested ${result.detections.length} plate box${result.detections.length === 1 ? '' : 'es'}. Verify each guess; the saved annotation will include both the guess and confirmed text.`,
          false,
        );
      } else {
        showAlert(
          boxes.length
            ? 'This run found no new plates. Kept the boxes already shown.'
            : 'AI found no plates. Draw boxes manually to add plate annotations.',
          false,
        );
      }
    } catch (error) {
      if (generation === imageGeneration) showAlert(error.message || 'AI detection failed.');
    } finally {
      detecting = false;
      detectButton.textContent = 'Auto-detect';
      updateSaveState();
    }
  });

  if (batchButton && rawSelect) {
    batchButton.addEventListener('click', async () => {
      if (batchDetecting) {
        stopBatchRequested = true;
        batchButton.textContent = 'Stopping…';
        batchButton.disabled = true;
        return;
      }
      if (detecting) return;

      const filenames = Array.from(rawSelect.options, option => option.value);
      if (!filenames.length) return;
      batchDetecting = true;
      stopBatchRequested = false;
      batchStatus.hidden = false;
      batchButton.textContent = 'Stop batch';
      updateSaveState();
      let detected = 0;
      let noPlates = 0;
      let failed = 0;
      let firstFailure = '';
      let completed = 0;

      for (const filename of filenames) {
        if (stopBatchRequested) break;
        batchStatus.textContent = `Detecting ${completed + 1} of ${filenames.length}: ${filename}`;
        try {
          // Always rerun inference: an existing record may contain only a
          // partial set of plates, so its mere presence cannot mean “complete”.
          const form = new FormData();
          form.append('source', 'rawimages');
          form.append('rawimage', filename);
          const detectResponse = await fetch('/api/label-detections', { method: 'POST', body: form });
          const result = await detectResponse.json();
          if (!detectResponse.ok) throw new Error(result.error || 'AI detection failed.');
          detected += 1;
          if (!Array.isArray(result.detections) || result.detections.length === 0) noPlates += 1;
          if (sourceType === 'rawimages' && selectedRawImage === filename) {
            await restoreSavedImageInfo(imageGeneration, filename);
          }
        } catch (error) {
          failed += 1;
          if (!firstFailure) firstFailure = error.message || 'Unknown error';
        }
        completed += 1;
      }

      const stopped = stopBatchRequested;
      batchDetecting = false;
      batchButton.textContent = 'Detect all images';
      if (sourceType === 'rawimages' && selectedRawImage) {
        await restoreSavedImageInfo(imageGeneration, selectedRawImage);
      }
      const summary = `${detected} processed · ${noPlates} with no new boxes returned · ${failed} failed`;
      batchStatus.textContent = `${stopped ? 'Stopped' : 'Finished'}: ${summary}${stopped ? ` · ${completed} of ${filenames.length} checked` : ''}${firstFailure ? ` · First error: ${firstFailure}` : ''}`;
      updateSaveState();
    });
  }

  canvas.addEventListener('pointerdown', event => {
    if (!imageReady) return;
    canvas.setPointerCapture(event.pointerId);
    const point = canvasPoint(event);
    drawing = { x1: point.x, y1: point.y, x2: point.x, y2: point.y };
    draw();
  });
  canvas.addEventListener('pointermove', event => {
    if (!drawing) return;
    const point = canvasPoint(event);
    drawing.x2 = Math.max(0, Math.min(canvas.width, point.x));
    drawing.y2 = Math.max(0, Math.min(canvas.height, point.y));
    draw();
  });
  canvas.addEventListener('pointerup', () => {
    if (!drawing) return;
    if (Math.abs(drawing.x2 - drawing.x1) > 8 && Math.abs(drawing.y2 - drawing.y1) > 8) {
      userEditedImage = true;
      boxes.push({ ...drawing, text: '' });
      renderObjectList();
    }
    drawing = null;
    draw();
  });

  canvas.addEventListener('contextmenu', event => {
    event.preventDefault();
    if (drawing) return;
    const point = canvasPoint(event);
    for (let index = boxes.length - 1; index >= 0; index -= 1) {
      const box = boxes[index];
      const left = Math.min(box.x1, box.x2);
      const right = Math.max(box.x1, box.x2);
      const top = Math.min(box.y1, box.y2);
      const bottom = Math.max(box.y1, box.y2);
      if (point.x >= left && point.x <= right && point.y >= top && point.y <= bottom) {
        userEditedImage = true;
        boxes.splice(index, 1);
        renderObjectList();
        draw();
        return;
      }
    }
  });

  document.getElementById('clear-boxes').addEventListener('click', () => {
    userEditedImage = true;
    boxes = [];
    renderObjectList();
    draw();
  });

  saveButton.addEventListener('click', async () => {
    if (!imageReady || !boxes.length) return;
    alertBox.hidden = true;
    savedMessage.hidden = true;
    saveButton.disabled = true;
    saveButton.querySelector('span:first-child').textContent = 'Saving annotation…';
    const scaleX = image.naturalWidth / canvas.width;
    const scaleY = image.naturalHeight / canvas.height;
    const annotations = boxes.map(box => ({
      bbox: [box.x1 * scaleX, box.y1 * scaleY, box.x2 * scaleX, box.y2 * scaleY],
      text: box.text.trim(),
    }));
    const form = new FormData();
    form.append('source', sourceType);
    if (sourceType === 'upload') form.append('image', imageFile, imageFile.name);
    else form.append('rawimage', selectedRawImage);
    form.append('annotations', JSON.stringify(annotations));
    try {
      const response = await fetch('/api/labels', { method: 'POST', body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not save labels.');
      const saveStatus = result.updated ? 'Updated the existing label for this image.' : 'Saved to this server.';
      savedMessage.innerHTML = `${saveStatus} <a href="${result.json_url}">Download annotation JSON ↗</a><span>Each plate object includes coordinates, the AI OCR guess, and the verified OCR text.</span>`;
      savedMessage.hidden = false;
    } catch (error) {
      showAlert(error.message || 'Could not save labels.');
    } finally {
      saveButton.querySelector('span:first-child').textContent = 'Save labels as JSON';
      updateSaveState();
    }
  });

  document.querySelectorAll('input[name="label-source"]').forEach(option => {
    if (option.checked && option.value === 'rawimages') updateSourceMode();
  });
})();
