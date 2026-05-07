# tests/conftest.py
import pytest
from devcontext.config.settings import setup_tracing

@pytest.fixture(autouse=True, scope="session")
def setup():
    """Run once before all tests — initialize tracing."""
    setup_tracing()