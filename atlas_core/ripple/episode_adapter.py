"""Episode -> Ripple adapter.

`AtlasGraphiti.add_episode` produces a set of Graphiti `EntityEdge`s. The Ripple
engine, by contrast, cascades one *belief confidence change at a time* via
`RippleEngine.propagate(upstream_kref, *, old_confidence, new_confidence)`. The
two vocabularies never lined up: the ingestion hook was written against a
`propagate(new_edges=..., invalidated_edges=..., episode=...)` signature that the
engine has never exposed, so the flagship composition raised `TypeError` the
instant `ripple_engine` + `ledger` were wired.

This module is the missing translation layer. It maps the per-episode edge set
onto a deterministic list of per-kref confidence changes that `propagate()`
actually accepts.

Contract
--------
Atlas belief metadata rides on the Graphiti edge `attributes` dict — the same
storage convention the ontology layer documents for entity types
(see `atlas_core/ontology/base.py`). An edge participates in a cascade only when
it carries a parseable `kref`; structural / non-belief edges are skipped. Per
edge:

  * `kref`              -> the upstream belief whose confidence moved.
  * `confidence_score`  -> the belief's confidence AFTER ingestion.
  * `prior_confidence`  -> the belief's confidence BEFORE (optional; defaults to
                           `confidence_score`, i.e. a no-op delta, when absent).

An *invalidated* edge (Graphiti set `expired_at`) is a retraction: its belief
collapses to `new_confidence = 0.0`. When the same kref appears as both promoted
and invalidated in one episode, the retraction wins.

Spec: notes/06 - Ripple Algorithm Spec.md § 4 (promotion-gated cascade)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from atlas_core.revision.uri import Kref, KrefParseError

if TYPE_CHECKING:
    from graphiti_core.edges import EntityEdge

log = logging.getLogger(__name__)

# Confidence a retracted (invalidated) belief collapses to.
_RETRACTED_CONFIDENCE = 0.0


@dataclass(frozen=True)
class BeliefConfidenceChange:
    """One upstream belief confidence change, shaped for `propagate()`."""

    upstream_kref: str
    old_confidence: float
    new_confidence: float
    belief_text: str = ""


def _edge_kref(edge: EntityEdge) -> str | None:
    """Return the belief kref carried by an edge, or None if it carries none
    (structural / non-belief edge) or an unparseable one."""
    attrs = getattr(edge, "attributes", None) or {}
    raw = attrs.get("kref")
    if not raw:
        return None
    try:
        Kref.parse(raw)
    except KrefParseError:
        log.debug("Skipping edge %s: unparseable kref %r", edge.uuid, raw)
        return None
    return raw


def _confidence(edge: EntityEdge, key: str, default: float) -> float:
    attrs = getattr(edge, "attributes", None) or {}
    value = attrs.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        log.debug(
            "Edge %s: non-numeric %s=%r; using default %s",
            edge.uuid, key, value, default,
        )
        return default


def episode_edges_to_changes(
    promoted_edges: list[EntityEdge],
    invalidated_edges: list[EntityEdge],
) -> list[BeliefConfidenceChange]:
    """Translate an episode's promoted + invalidated edges into per-kref
    confidence changes for `RippleEngine.propagate`.

    Deterministic, side-effect-free, and does not touch Neo4j. Deduplicates by
    kref; a retraction (invalidated edge) always overrides a promotion for the
    same kref.

    Args:
        promoted_edges: Edges promoted to the trust ledger this episode.
        invalidated_edges: Edges Graphiti expired this episode.

    Returns:
        One `BeliefConfidenceChange` per distinct belief kref, in stable order
        (promotions first in first-seen order, then retractions).
    """
    changes: dict[str, BeliefConfidenceChange] = {}

    for edge in promoted_edges:
        kref = _edge_kref(edge)
        if kref is None:
            continue
        new_conf = _confidence(edge, "confidence_score", 1.0)
        old_conf = _confidence(edge, "prior_confidence", new_conf)
        changes[kref] = BeliefConfidenceChange(
            upstream_kref=kref,
            old_confidence=old_conf,
            new_confidence=new_conf,
            belief_text=edge.fact or "",
        )

    # Retractions override: an invalidated belief collapses to zero confidence.
    for edge in invalidated_edges:
        kref = _edge_kref(edge)
        if kref is None:
            continue
        old_conf = _confidence(
            edge, "prior_confidence",
            _confidence(edge, "confidence_score", 1.0),
        )
        changes[kref] = BeliefConfidenceChange(
            upstream_kref=kref,
            old_confidence=old_conf,
            new_confidence=_RETRACTED_CONFIDENCE,
            belief_text=edge.fact or "",
        )

    return list(changes.values())
