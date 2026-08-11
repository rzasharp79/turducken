"""Self-documenting configuration fields.

The premise of turducken is that a training recipe should be able to explain
itself. Rather than keeping settings in one place and documentation in another
(where they drift apart), every tunable field is declared with the pedagogy
attached: what it does in plain language, why it matters, what you trade away
by moving it, and how advanced it is.

The UI reads this metadata straight off the JSON schema, so a new knob shows up
in the interface fully explained the moment it is declared. There is no second
place to update.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

NAMESPACE = "turducken"


class Tier(str, Enum):
    """How much rope a knob gives you, and therefore when to show it.

    The UI reveals tiers progressively: a first-time user sees only ESSENTIAL
    and never has to scroll past a wall of settings they cannot yet evaluate.
    """

    ESSENTIAL = "essential"
    """You must decide this. There is no sensible universal default."""

    STANDARD = "standard"
    """Worth understanding on your second or third run. Defaults are decent."""

    ADVANCED = "advanced"
    """Defaults are good. Change only with a specific reason you can articulate."""

    EXPERIMENTAL = "experimental"
    """Sharp edges. May interact badly with other settings."""


class KnobDoc(BaseModel):
    """The teaching material attached to a single setting."""

    plain: str
    """One sentence, no jargon. What is this?"""

    why: str
    """Why this setting exists and what it actually changes during training."""

    tradeoff: str = ""
    """What you give up by turning it up, and what you give up by turning it down."""

    tier: Tier = Tier.STANDARD
    unit: str | None = None
    """Human-facing unit, e.g. 'steps', 'px', 'GB'."""

    lesson: str | None = None
    """Slug of an explainer module that covers this in depth."""

    rule_of_thumb: str = ""
    """A concrete starting value and the reasoning behind it."""

    symptoms: dict[str, str] = Field(default_factory=dict)
    """Observable failure modes, e.g. {"too high": "...", "too low": "..."}.

    This is the part people actually need. Knowing what a knob means is not the
    same as recognising, in your own results, that it is set wrong.
    """


def knob(
    default: Any,
    *,
    plain: str,
    why: str,
    tradeoff: str = "",
    tier: Tier = Tier.STANDARD,
    unit: str | None = None,
    lesson: str | None = None,
    rule_of_thumb: str = "",
    symptoms: dict[str, str] | None = None,
    **field_kwargs: Any,
) -> Any:
    """Declare a documented settings field.

    Behaves exactly like ``pydantic.Field`` — including validation constraints
    such as ``ge``/``le`` — but stashes the documentation in the field's JSON
    schema under the ``turducken`` key so the API and UI can render it.
    """
    doc = KnobDoc(
        plain=plain,
        why=why,
        tradeoff=tradeoff,
        tier=tier,
        unit=unit,
        lesson=lesson,
        rule_of_thumb=rule_of_thumb,
        symptoms=symptoms or {},
    )
    extra = field_kwargs.pop("json_schema_extra", {}) or {}
    extra[NAMESPACE] = doc.model_dump(exclude_none=True)
    return Field(default, json_schema_extra=extra, description=plain, **field_kwargs)


def _walk(schema: dict, defs: dict, prefix: str = "") -> dict[str, dict]:
    """Recursively collect documented fields out of a JSON schema."""
    found: dict[str, dict] = {}
    for name, prop in (schema.get("properties") or {}).items():
        path = f"{prefix}{name}"
        if NAMESPACE in prop:
            found[path] = prop[NAMESPACE]
        # Follow $ref into nested models so grouped settings are discoverable.
        ref = prop.get("$ref") or next(
            (a.get("$ref") for a in prop.get("allOf", []) if "$ref" in a), None
        )
        if ref:
            target = defs.get(ref.rsplit("/", 1)[-1])
            if target and target.get("properties"):
                found.update(_walk(target, defs, prefix=f"{path}."))
    return found


def collect_docs(model: type[BaseModel]) -> dict[str, dict]:
    """Return ``{dotted.field.path: KnobDoc-as-dict}`` for a model.

    Used by the API to hand the UI everything it needs to render an annotated
    form, and by the test suite to assert that no knob ships undocumented.
    """
    schema = model.model_json_schema(ref_template="#/$defs/{model}")
    return _walk(schema, schema.get("$defs", {}))
