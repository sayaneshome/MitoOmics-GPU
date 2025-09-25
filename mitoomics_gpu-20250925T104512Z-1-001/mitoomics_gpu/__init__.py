"""MitoOmics-GPU package init (safe imports; no hard API assumptions)."""

__version__ = "0.1.0"

# Optional convenience export for CLI
try:
    from .cli import main as cli_main  # noqa: F401
except Exception:
    def cli_main(*args, **kwargs):  # noqa: D401
        """CLI entrypoint not available; run `python -m mitoomics_gpu.cli` instead."""
        raise SystemExit("Use: python -m mitoomics_gpu.cli ...")

# Legacy compatibility: expose compute_mhi if it exists, otherwise provide a clear error.
try:
    from .mhi import compute_mhi  # type: ignore  # noqa: F401
except Exception:
    def compute_mhi(*args, **kwargs):
        raise ImportError(
            "compute_mhi is not exported by mitoomics_gpu.mhi. "
            "Use the CLI (`python -m mitoomics_gpu.cli ...`) or import functions from mitoomics_gpu.mhi directly."
        )
