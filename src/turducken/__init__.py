"""turducken — a user-friendly, low-overhead LoRA training solution.

The app is built around one idea: a sophisticated tool does not have to be an
opaque one. Every setting explains itself, every recommendation shows its
reasoning, and the optional explainer modules go all the way down to the
mathematics — without ever standing between you and starting a run.
"""

__version__ = "0.1.0"

from .advisor import review
from .schemas.recipe import Recipe

__all__ = ["Recipe", "review", "__version__"]
