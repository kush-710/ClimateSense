"""Unit tests for the assistant's pure-logic city detection (no network, no DB)."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.assistant import detect_city


def test_detect_city_delhi():
    assert detect_city("Is it safe to run in Delhi today?") == 1


def test_detect_city_bengaluru_alias():
    assert detect_city("How's the AQI in Bangalore?") == 2


def test_detect_city_none_when_unspecified():
    assert detect_city("Is it safe to go for a run right now?") is None
