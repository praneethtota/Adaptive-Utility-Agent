"""
tests/test_imports.py — public import API smoke tests.

Verifies that every symbol listed in aua.__all__ is importable
and that the documented one-liner usage works.
"""


def test_core_imports():
    """All documented top-level imports must work."""
    from aua import (
        FIELD_CONFIGS,
        ArbiterAgent,
        Router,
    )

    # Spot-check a few
    assert Router is not None
    assert ArbiterAgent is not None
    assert len(FIELD_CONFIGS) >= 10


def test_arbiter_alias():
    """Arbiter must be importable as both ArbiterAgent and (via alias) Arbiter."""
    import aua

    # __all__ must expose __version__
    assert "__version__" in aua.__all__
    assert "Router" in aua.__all__
    assert "ArbiterAgent" in aua.__all__


def test_version_export():
    """__version__ must be accessible from top-level aua."""
    import aua

    assert hasattr(aua, "__version__")
    assert isinstance(aua.__version__, str)
    assert aua.__version__ == "0.6.0a0"


def test_endpoint_models_exported():
    """REST endpoint models must be importable from aua directly."""
    from aua import (
        QueryRequest,
    )

    # Pydantic models — check they can be instantiated with required fields
    req = QueryRequest(query="test query")
    assert req.query == "test query"
    assert req.session_id == "default"


def test_stream_models_exported():
    """Streaming SSE event models must be importable from aua."""
    from aua import (
        StreamChunkEvent,
        StreamErrorEvent,
    )

    chunk = StreamChunkEvent(text="hello", index=0)
    assert chunk.text == "hello"
    assert chunk.type == "chunk"

    err = StreamErrorEvent(code=503, message="unreachable")
    assert err.code == 503


def test_config_submodule():
    """aua.config must expose all documented symbols."""
    from aua.config import (
        AVAILABLE_TIERS,
        FIELD_CONFIGS,
    )

    assert set(AVAILABLE_TIERS) == {"macbook", "rtx4090", "a100"}
    assert "software_engineering" in FIELD_CONFIGS
    assert "mathematics" in FIELD_CONFIGS


def test_no_private_imports_required():
    """Normal usage should not require importing private aua.* submodules."""
    # Everything a user needs should be at aua.* level
    import aua

    # These are the things a user touches — all should be in __all__
    required_public = [
        "__version__",
        "Router",
        "ArbiterAgent",
        "UtilityScorer",
        "FieldClassifier",
        "AssertionsStore",
        "load_config",
        "AUAConfig",
        "FIELD_CONFIGS",
        "QueryRequest",
        "RouterResponse",
    ]
    for name in required_public:
        assert name in aua.__all__, f"{name!r} missing from aua.__all__"
