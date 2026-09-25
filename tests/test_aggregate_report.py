"""
tests/test_aggregate_report.py
==============================
Unit and scenario tests for backend/aggregate_report.py.

Tests the location classification logic, event aggregation, and report
formatting matching requirements_new.md.

Run with:
    python -m pytest tests/test_aggregate_report.py -v
or:
    python3 tests/test_aggregate_report.py
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

# Make sure the project root is on sys.path when running from any directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pytest
except ImportError:
    # Minimal fallback fixture decorator so the file can also run directly with python3
    class _MockPytest:
        @staticmethod
        def fixture(fn):
            return fn
    pytest = _MockPytest()

from backend.aggregate_report import (
    CATEGORY_MAP,
    aggregate_events,
    classify_location,
    format_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def industrial_events() -> List[Dict[str, Any]]:
    """Clearly Industrial scenario: traffic + construction covering >60% of a 300s file.

    Duration breakdown (300 s total):
      - traffic:      00:00 - 02:00 (120 s = 40.0%)
      - construction: 01:40 - 03:20 (100 s = 33.33%)
      - bird_chirp:   00:40 - 01:10 ( 30 s = 10.0%)
      -> Total industrial: 120s + 100s = 220s (73.33% > 60%)
    """
    return [
        {
            "label": "traffic",
            "start": 0.0,
            "end": 120.0,
            "confidence": 0.92,
            "similar_to": None,
        },
        {
            "label": "construction",
            "start": 100.0,
            "end": 200.0,
            "confidence": 0.88,
            "similar_to": None,
        },
        {
            "label": "bird_chirp",
            "start": 40.0,
            "end": 70.0,
            "confidence": 0.75,
            "similar_to": None,
        },
    ]


@pytest.fixture
def natural_events() -> List[Dict[str, Any]]:
    """Clearly Natural habitat scenario: bird + wind covering >60% of a 300s file.

    Duration breakdown (300 s total):
      - bird:    00:00 - 01:50 (110 s = 36.67%)
      - wind:    01:30 - 03:30 (120 s = 40.0%)
      - traffic: 03:20 - 03:50 ( 30 s = 10.0%)
      -> Total natural: 110s + 120s = 230s (76.67% > 60%)
    """
    return [
        {
            "label": "bird",
            "start": 0.0,
            "end": 110.0,
            "confidence": 0.89,
            "similar_to": None,
        },
        {
            "label": "wind",
            "start": 90.0,
            "end": 210.0,
            "confidence": 0.85,
            "similar_to": None,
        },
        {
            "label": "traffic",
            "start": 200.0,
            "end": 230.0,
            "confidence": 0.70,
            "similar_to": None,
        },
    ]


@pytest.fixture
def mixed_events() -> List[Dict[str, Any]]:
    """Mixed scenario with no dominant category, including an 'Unknown' event.

    Duration breakdown (300 s total):
      - traffic:  00:10 - 01:10 (60 s = 20.0%)
      - bird:     01:20 - 02:20 (60 s = 20.0%)
      - Unknown:  02:30 - 03:10 (40 s = 13.33%, with similar_to top-3 matches)
      - drilling: 03:40 - 04:20 (40 s = 13.33%)
      -> Industrial: 33.33%, Natural: 20.0%, Unknown/Other: 13.33% (none > 60%)
    """
    return [
        {
            "label": "traffic",
            "start": 10.0,
            "end": 70.0,
            "confidence": 0.82,
            "similar_to": None,
        },
        {
            "label": "bird",
            "start": 80.0,
            "end": 140.0,
            "confidence": 0.87,
            "similar_to": None,
        },
        {
            "label": "Unknown",
            "start": 150.0,
            "end": 190.0,
            "confidence": 0.40,
            "similar_to": [
                {"label": "drilling", "similarity": 0.65},
                {"label": "engine_idling", "similarity": 0.25},
                {"label": "jackhammer", "similarity": 0.10},
            ],
        },
        {
            "label": "drilling",
            "start": 220.0,
            "end": 260.0,
            "confidence": 0.78,
            "similar_to": None,
        },
    ]


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestAggregateReportScenarios:
    """Scenario tests verifying event aggregation, location classification, and reporting."""

    TOTAL_DURATION: float = 300.0

    def test_industrial_scenario(self, industrial_events: List[Dict[str, Any]]):
        """Traffic and construction covering >60% must be classified as 'Industrial'."""
        agg = aggregate_events(industrial_events, total_duration=self.TOTAL_DURATION)
        report = format_report(industrial_events, agg)

        print("\n--- Industrial Scenario Report ---")
        print(report)

        # Assertions on aggregation result
        assert agg["location"] == "Industrial", (
            f"Expected 'Industrial', got {agg['location']!r}"
        )
        assert agg["dominant_category"] == "industrial"
        assert agg["dominant_pct"] > 60.0

        # Assertions on formatted text report
        assert "1. Detected sounds:" in report
        assert "2. Location classification: Industrial" in report
        assert "Traffic" in report
        assert "Construction" in report
        assert "confidence: 0.92" in report

    def test_natural_habitat_scenario(self, natural_events: List[Dict[str, Any]]):
        """Bird and wind covering >60% must be classified as 'Natural habitat'."""
        agg = aggregate_events(natural_events, total_duration=self.TOTAL_DURATION)
        report = format_report(natural_events, agg)

        print("\n--- Natural Habitat Scenario Report ---")
        print(report)

        # Assertions on aggregation result
        assert agg["location"] == "Natural habitat", (
            f"Expected 'Natural habitat', got {agg['location']!r}"
        )
        assert agg["dominant_category"] == "natural"
        assert agg["dominant_pct"] > 60.0

        # Assertions on formatted text report
        assert "1. Detected sounds:" in report
        assert "2. Location classification: Natural habitat" in report
        assert "Bird" in report
        assert "Wind" in report
        assert "confidence: 0.89" in report

    def test_mixed_scenario(self, mixed_events: List[Dict[str, Any]]):
        """No single category covering >60% must be classified as 'Mixed / Residential'."""
        agg = aggregate_events(mixed_events, total_duration=self.TOTAL_DURATION)
        report = format_report(mixed_events, agg)

        print("\n--- Mixed Scenario Report ---")
        print(report)

        # Assertions on aggregation result
        assert agg["location"] == "Mixed / Residential", (
            f"Expected 'Mixed / Residential', got {agg['location']!r}"
        )

        # Assertions on formatted text report
        assert "1. Detected sounds:" in report
        assert "2. Location classification: Mixed / Residential" in report
        assert "Unknown" in report
        assert "closest matches: 65% drilling, 25% engine idling, 10% jackhammer" in report


class TestClassifyLocationThresholds:
    """Direct boundary tests for the 60% decision threshold in requirements_new.md."""

    def test_strictly_greater_than_60_required(self):
        """Exactly 60.0% must NOT trigger Industrial; strictly > 60% is required."""
        assert classify_location({"industrial": 60.0}) == "Mixed / Residential"
        assert classify_location({"industrial": 60.1}) == "Industrial"

    def test_natural_habitat_threshold(self):
        """Exactly 60.0% must NOT trigger Natural habitat; strictly > 60% is required."""
        assert classify_location({"natural": 60.0}) == "Mixed / Residential"
        assert classify_location({"natural": 60.1}) == "Natural habitat"

    def test_empty_or_zero_percentages(self):
        """Empty input must safely default to 'Mixed / Residential'."""
        assert classify_location({}) == "Mixed / Residential"
        assert classify_location({"industrial": 0.0, "natural": 0.0}) == "Mixed / Residential"


# ---------------------------------------------------------------------------
# Direct Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tester = TestAggregateReportScenarios()

    print("Running test_industrial_scenario...")
    tester.test_industrial_scenario(industrial_events())

    print("Running test_natural_habitat_scenario...")
    tester.test_natural_habitat_scenario(natural_events())

    print("Running test_mixed_scenario...")
    tester.test_mixed_scenario(mixed_events())

    threshold_tester = TestClassifyLocationThresholds()
    threshold_tester.test_strictly_greater_than_60_required()
    threshold_tester.test_natural_habitat_threshold()
    threshold_tester.test_empty_or_zero_percentages()

    print("\n All aggregate report scenario tests passed successfully!")
