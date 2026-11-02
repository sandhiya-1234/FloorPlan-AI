"""
FloorPlan-AI: Pydantic Data Schema for 2D Architectural Floor Plan Detection.

Validates architectural elements (walls, doors, windows, rooms), coordinates,
and spatial consistency for downstream 2D-to-3D CAD pipelines.
"""

from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class OrientationEnum(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    SLANTED = "slanted"
    UNKNOWN = "unknown"


class RoomTypeEnum(str, Enum):
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    DINING_ROOM = "dining_room"
    CORRIDOR = "corridor"
    BALCONY = "balcony"
    STORAGE = "storage"
    OFFICE = "office"
    UNDIVIDED = "undivided"
    OTHER = "other"
    UNKNOWN = "unknown"


class RoomElement(BaseModel):
    """Architectural room polygon representation."""
    id: str = Field(..., description="Unique room identifier, e.g. room_1")
    type: str = Field(..., description="Room functional category")
    polygon: List[List[float]] = Field(
        ...,
        description="List of [x, y] coordinates forming a closed or sequence polygon",
        min_length=3
    )
    area_pixels: Optional[float] = Field(None, description="Calculated area of room polygon in px^2")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence reported by the system; None when it reports none (never invented)")

    @field_validator("polygon")
    @classmethod
    def validate_polygon_vertices(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        for pt in v:
            if len(pt) != 2:
                raise ValueError(f"Each polygon vertex must be [x, y], got {pt}")
        return v


class WallElement(BaseModel):
    """Wall segment representation with endpoints and thickness."""
    id: str = Field(..., description="Unique wall identifier, e.g. wall_1")
    start: List[float] = Field(..., description="[x, y] start coordinate")
    end: List[float] = Field(..., description="[x, y] end coordinate")
    thickness: Optional[float] = Field(None, ge=1.0, description="Wall thickness in pixels; None when not known (no default is assumed)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence reported by the system; None when it reports none (never invented)")

    @field_validator("start", "end")
    @classmethod
    def validate_coord_len(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError(f"Coordinate must be a 2D [x, y] pair, got {v}")
        return v

    @model_validator(mode="after")
    def validate_non_zero_length(self) -> "WallElement":
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length < 1e-3:
            raise ValueError(f"Wall {self.id} start and end points cannot be identical (length={length:.2f})")
        return self


class DoorElement(BaseModel):
    """Architectural door representation."""
    id: str = Field(..., description="Unique door identifier, e.g. door_1")
    bbox: List[float] = Field(..., description="Bounding box [x_min, y_min, x_max, y_max]")
    position: Optional[List[float]] = Field(None, description="Center coordinate [x, y]")
    orientation: Optional[OrientationEnum] = Field(OrientationEnum.HORIZONTAL, description="Door orientation")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence reported by the system; None when it reports none (never invented)")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError(f"Bounding box must contain 4 values [xmin, ymin, xmax, ymax], got {v}")
        xmin, ymin, xmax, ymax = v
        if xmax < xmin or ymax < ymin:
            raise ValueError(f"Invalid bbox dimensions: [{xmin}, {ymin}, {xmax}, {ymax}]")
        return v

    @model_validator(mode="after")
    def compute_position_if_missing(self) -> "DoorElement":
        if self.position is None and len(self.bbox) == 4:
            xmin, ymin, xmax, ymax = self.bbox
            self.position = [round((xmin + xmax) / 2.0, 2), round((ymin + ymax) / 2.0, 2)]
        return self


class WindowElement(BaseModel):
    """Architectural window representation."""
    id: str = Field(..., description="Unique window identifier, e.g. window_1")
    bbox: List[float] = Field(..., description="Bounding box [x_min, y_min, x_max, y_max]")
    position: Optional[List[float]] = Field(None, description="Center coordinate [x, y]")
    orientation: Optional[OrientationEnum] = Field(OrientationEnum.HORIZONTAL, description="Window orientation")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence reported by the system; None when it reports none (never invented)")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError(f"Bounding box must contain 4 values [xmin, ymin, xmax, ymax], got {v}")
        xmin, ymin, xmax, ymax = v
        if xmax < xmin or ymax < ymin:
            raise ValueError(f"Invalid bbox dimensions: [{xmin}, {ymin}, {xmax}, {ymax}]")
        return v

    @model_validator(mode="after")
    def compute_position_if_missing(self) -> "WindowElement":
        if self.position is None and len(self.bbox) == 4:
            xmin, ymin, xmax, ymax = self.bbox
            self.position = [round((xmin + xmax) / 2.0, 2), round((ymin + ymax) / 2.0, 2)]
        return self


class FloorPlanSchema(BaseModel):
    """
    Standardized Architectural Floor Plan JSON Schema.
    Strictly validated for downstream 2D-to-3D CAD modeling.
    """
    image: str = Field(..., description="Filename or relative path of the source floor plan image")
    image_width: int = Field(..., gt=0, description="Width of image in pixels")
    image_height: int = Field(..., gt=0, description="Height of image in pixels")
    units: str = Field("pixels", description="Coordinate unit measurement")

    rooms: List[RoomElement] = Field(default_factory=list, description="Detected room entities")
    walls: List[WallElement] = Field(default_factory=list, description="Detected wall segment entities")
    doors: List[DoorElement] = Field(default_factory=list, description="Detected door entities")
    windows: List[WindowElement] = Field(default_factory=list, description="Detected window entities")

    @model_validator(mode="after")
    def validate_spatial_and_ids(self) -> "FloorPlanSchema":
        w = float(self.image_width)
        h = float(self.image_height)

        # 1. Enforce strict ID uniqueness across all architectural elements
        seen_ids = set()
        for category, items in [("room", self.rooms), ("wall", self.walls), ("door", self.doors), ("window", self.windows)]:
            for item in items:
                if item.id in seen_ids:
                    raise ValueError(f"Duplicate identifier '{item.id}' detected in {category} list. All element IDs must be unique.")
                seen_ids.add(item.id)

        # 2. Check room polygons within bounds & auto-calculate area if missing
        for room in self.rooms:
            clamped_poly = []
            for pt in room.polygon:
                x = max(0.0, min(w, float(pt[0])))
                y = max(0.0, min(h, float(pt[1])))
                clamped_poly.append([round(x, 2), round(y, 2)])
            room.polygon = clamped_poly

            if room.area_pixels is None or room.area_pixels <= 0:
                # Shoelace formula for polygon area
                n = len(room.polygon)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += room.polygon[i][0] * room.polygon[j][1]
                    area -= room.polygon[j][0] * room.polygon[i][1]
                room.area_pixels = round(abs(area) / 2.0, 2)

        # 3. Check walls within bounds
        for wall in self.walls:
            wall.start = [
                round(max(0.0, min(w, float(wall.start[0]))), 2),
                round(max(0.0, min(h, float(wall.start[1]))), 2)
            ]
            wall.end = [
                round(max(0.0, min(w, float(wall.end[0]))), 2),
                round(max(0.0, min(h, float(wall.end[1]))), 2)
            ]

        # 4. Check doors & windows bbox within bounds
        for item in self.doors + self.windows:
            xmin, ymin, xmax, ymax = item.bbox
            item.bbox = [
                round(max(0.0, min(w, float(xmin))), 2),
                round(max(0.0, min(h, float(ymin))), 2),
                round(max(0.0, min(w, float(xmax))), 2),
                round(max(0.0, min(h, float(ymax))), 2)
            ]
            if item.position:
                item.position = [
                    round(max(0.0, min(w, float(item.position[0]))), 2),
                    round(max(0.0, min(h, float(item.position[1]))), 2)
                ]

        return self

    def to_compact_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary formatted for model training targets."""
        return self.model_dump(exclude_none=True)
