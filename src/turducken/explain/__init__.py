"""The optional teaching layer: lessons and runnable demonstrations."""

from .demos import DEMOS, run_demo
from .modules import LESSONS, Lesson, get_lesson, list_lessons

__all__ = ["DEMOS", "LESSONS", "Lesson", "get_lesson", "list_lessons", "run_demo"]
