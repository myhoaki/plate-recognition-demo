"""Small CRNN used for local license-plate OCR fine-tuning and inference."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
from fast_alpr.alpr import BaseOCR, OcrResult


PLATE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OCR_HEIGHT = 48
OCR_WIDTH = 160


class PlateOCRNetwork(nn.Module):
    def __init__(self, class_count: int = len(PLATE_ALPHABET) + 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),
        )
        self.sequence = nn.LSTM(128, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.classifier = nn.Linear(256, class_count)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).mean(dim=2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)


def clean_plate_text(text: str) -> str:
    return "".join(char for char in text.upper() if char in PLATE_ALPHABET)


def prepare_plate_crop(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Plate crop is empty")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (OCR_WIDTH, OCR_HEIGHT), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[None, :, :]


def decode_greedy(logits: torch.Tensor) -> tuple[str, float]:
    probabilities = logits.softmax(dim=-1)
    scores, indices = probabilities.max(dim=-1)
    blank = len(PLATE_ALPHABET)
    text = []
    last = blank
    confidences = []
    for index, score in zip(indices.tolist(), scores.tolist()):
        if index != blank and index != last:
            text.append(PLATE_ALPHABET[index])
            confidences.append(score)
        last = index
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return "".join(text), confidence


class TrainedPlateOCR(BaseOCR):
    """Adapter for fast-alpr's BaseOCR protocol."""

    def __init__(self, model_path: str):
        super().__init__()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = PlateOCRNetwork()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, cropped_plate: np.ndarray):
        tensor = torch.from_numpy(prepare_plate_crop(cropped_plate)).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(tensor)[0]
        text, confidence = decode_greedy(logits)
        return OcrResult(text=text, confidence=confidence)
