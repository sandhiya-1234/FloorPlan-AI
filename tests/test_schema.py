"""
Unit tests for FloorPlanSchema and Pydantic validation.
"""

import pytest
from pydantic import ValidationError
from schemas.floorplan_schema import (
    FloorPlanSchema, RoomElement, WallElement, DoorElement, WindowElement, OrientationEnum
)


def test_valid_floorplan_schema():
    """Tests building a completely valid schema instance."""
    schema = FloorPlanSchema(
        image="test_plan.png",
        image_width=1000,
        image_height=800,
        rooms=[
            RoomElement(
                id="room_1",
                type="bedroom",
                polygon=[[100, 100], [400, 100], [400, 400], [100, 400]]
            )
        ],
        walls=[
            WallElement(id="wall_1", start=[100, 100], end=[400, 100], thickness=10)
        ],
        doors=[
            DoorElement(id="door_1", bbox=[200, 95, 260, 105], orientation=OrientationEnum.HORIZONTAL)
        ],
        windows=[
            WindowElement(id="win_1", bbox=[300, 95, 380, 105], orientation=OrientationEnum.HORIZONTAL)
        ]
    )
    assert schema.image_width == 1000
    assert len(schema.rooms) == 1
    assert schema.rooms[0].area_pixels == 90000.0
    assert schema.doors[0].position == [230.0, 100.0]


def test_invalid_polygon_vertices():
    """Tests that a polygon with fewer than 3 vertices is rejected."""
    with pytest.raises(ValidationError):
        RoomElement(id="r_invalid", type="kitchen", polygon=[[100, 100], [200, 200]])


def test_zero_length_wall_rejected():
    """Tests that a wall with identical start and end points is rejected."""
    with pytest.raises(ValidationError):
        WallElement(id="w_invalid", start=[100, 100], end=[100, 100])


def test_invalid_bbox_rejected():
    """Tests that a bounding box with xmin > xmax is rejected."""
    with pytest.raises(ValidationError):
        DoorElement(id="d_invalid", bbox=[300, 100, 200, 150])


def test_coordinate_clamping():
    """Tests that coordinates beyond image bounds are clamped safely."""
    schema = FloorPlanSchema(
        image="out_of_bounds.png",
        image_width=500,
        image_height=500,
        rooms=[
            RoomElement(id="r1", type="living_room", polygon=[[-50, 0], [600, 0], [600, 600], [-50, 600]])
        ]
    )
    # Coordinates should be clamped to [0, 500]
    for pt in schema.rooms[0].polygon:
        assert 0.0 <= pt[0] <= 500.0
        assert 0.0 <= pt[1] <= 500.0


def test_duplicate_id_rejected():
    """Tests that duplicate element IDs trigger a validation error."""
    with pytest.raises(ValidationError):
        FloorPlanSchema(
            image="dup.png",
            image_width=500,
            image_height=500,
            walls=[
                WallElement(id="elem_1", start=[10, 10], end=[100, 10]),
                WallElement(id="elem_1", start=[100, 10], end=[200, 10]),
            ]
        )


def test_shoelace_area_orientation_invariant():
    """Tests that clockwise and counter-clockwise polygons yield positive area."""
    # Counter-clockwise
    r_ccw = RoomElement(id="r_ccw", type="bedroom", polygon=[[0, 0], [10, 0], [10, 10], [0, 10]])
    schema1 = FloorPlanSchema(image="t.png", image_width=100, image_height=100, rooms=[r_ccw])
    assert schema1.rooms[0].area_pixels == 100.0

    # Clockwise
    r_cw = RoomElement(id="r_cw", type="bedroom", polygon=[[0, 0], [0, 10], [10, 10], [10, 0]])
    schema2 = FloorPlanSchema(image="t.png", image_width=100, image_height=100, rooms=[r_cw])
    assert schema2.rooms[0].area_pixels == 100.0
