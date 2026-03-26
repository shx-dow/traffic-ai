from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.roi import parse_rect_roi


def test_parse_rect_roi_valid():
    assert parse_rect_roi("10,20,30,40") == (10, 20, 30, 40)


def test_parse_rect_roi_empty_returns_none():
    assert parse_rect_roi("") is None


def test_parse_rect_roi_invalid_raises():
    try:
        parse_rect_roi("1,2,3")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid roi format")


if __name__ == "__main__":
    test_parse_rect_roi_valid()
    test_parse_rect_roi_empty_returns_none()
    test_parse_rect_roi_invalid_raises()
    print("PASS test_roi")
