"""Public RFD4-Proteina API: ``import rfd4_proteina as rfd4``."""

from collections.abc import Sequence
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfproteina.proteina import Proteina

__version__ = version("rfd4-proteina")
__all__ = ["Proteina", "__version__", "generate"]


def __getattr__(name: str) -> Any:
    """Load the model on demand to avoid initializing Torch during package inspection."""
    if name == "Proteina":
        from rfproteina.proteina import Proteina

        globals()[name] = Proteina
        return Proteina
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate(model: "Proteina | None" = None, *, overrides: "Sequence[str]" = ()) -> None:
    """Compose ``inference.yaml`` with Hydra ``overrides`` and run generation.

    Args:
        model: A model loaded with ``Proteina.from_pretrained``; ``None`` loads the checkpoint named by the config.
        overrides: Hydra overrides, the same ``key=value`` strings that ``rfd4-generate`` accepts, for example
            ``["inputs=design.yaml", "out_dir=runs/first"]``.
    """
    import rootutils
    from hydra import compose, initialize_config_dir

    import rfproteina
    from rfproteina.generate import run_generation

    config_dir = rootutils.find_root(rfproteina.__file__, indicator=".project-root") / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="inference", overrides=list(overrides))
    run_generation(cfg, model=model)
