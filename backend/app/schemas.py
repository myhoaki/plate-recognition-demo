from pydantic import BaseModel, Field


class LabelInput(BaseModel):
    bbox_xyxy_pixels: list[float] = Field(min_length=4, max_length=4)
    verified_result: str = Field(default="", max_length=32)


class SaveAnnotations(BaseModel):
    annotations: list[LabelInput] = Field(max_length=100)


class StartDetection(BaseModel):
    image_id: str


class SetVerification(BaseModel):
    verified: bool
