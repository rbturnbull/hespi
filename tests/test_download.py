import tempfile
import pytest
from unittest.mock import patch
from pathlib import Path

from hespi import download

def test_get_cached_path():
    name = "tmpfile.txt"
    tmpfile = download.get_cached_path(name)
    assert tmpfile.parent.name == "hespi"
    assert tmpfile.parent.exists()
    assert tmpfile.name == name
