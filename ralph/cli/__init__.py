"""
Command-line interface for Ralph fault injection framework.

Provides entry points for running fault injection experiments.
"""

# Lazy import to avoid circular imports and runtime warnings
def main(*args, **kwargs):
    """Entry point for Ralph CLI. See ralph.cli.main for full docs."""
    from ralph.cli.main import main as _main
    return _main(*args, **kwargs)

__all__ = ["main"]
