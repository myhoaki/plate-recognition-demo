from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LabelInput(BaseModel):
    bbox_xyxy_pixels: list[float] = Field(min_length=4, max_length=4)
    verified_result: str = Field(default="", max_length=32)


class SaveAnnotations(BaseModel):
    annotations: list[LabelInput] = Field(max_length=100)


class OcrTextEntry(BaseModel):
    annotation_id: int
    verified_result: str = Field(default="", max_length=32)


class OcrTextUpdate(BaseModel):
    plates: list[OcrTextEntry] = Field(max_length=100)


class StartDetection(BaseModel):
    image_id: str


class RecognitionRequest(BaseModel):
    image_id: str
    detector_job_id: str | None = None
    ocr_job_id: str | None = None


class StartEvaluation(BaseModel):
    dataset_snapshot_id: str = Field(min_length=32, max_length=32)
    detector_job_id: str | None = None
    ocr_job_id: str | None = None


class UpdateOcrSample(BaseModel):
    plate_text: str = Field(default="", max_length=32)
    status: Literal["pending", "verified", "rejected"] = "pending"


class SetVerification(BaseModel):
    verified: bool


class SetReviewStatus(BaseModel):
    status: Literal["needs_review", "reviewed", "verified", "rejected", "no_plate"]


class StartTraining(BaseModel):
    model_type: Literal["detector", "ocr"] = "detector"
    dataset_snapshot_id: str = Field(min_length=32, max_length=32)
    epochs: int = Field(default=30, ge=1, le=300)
    batch_size: int = Field(default=8, ge=1, le=64)
    image_size: int = Field(default=640, ge=320, le=1280, multiple_of=32)
    device: str = Field(default="auto", pattern=r"^(auto|cpu|0|[1-9][0-9]*)$")
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    validation_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def ratios_sum_to_one(self):
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 0.001:
            raise ValueError("Train, validation, and test ratios must add up to 1.")
        return self


class ImportSnapshot(BaseModel):
    task: Literal["detector", "ocr"]
    path: str
