"""Unit tests for the episode -> Ripple adapter.

Pure translation layer: Graphiti `EntityEdge`s -> per-kref confidence changes
for `RippleEngine.propagate`. No Neo4j required.
"""

from datetime import datetime, timezone

import pytest

from atlas_core.ripple.episode_adapter import (
    BeliefConfidenceChange,
    episode_edges_to_changes,
)


def _edge(uuid_: str, *, kref=None, confidence=None, prior=None, fact="", expired=False):
    """Build a Graphiti EntityEdge carrying Atlas belief metadata in attributes."""
    from graphiti_core.edges import EntityEdge

    attrs: dict = {}
    if kref is not None:
        attrs["kref"] = kref
    if confidence is not None:
        attrs["confidence_score"] = confidence
    if prior is not None:
        attrs["prior_confidence"] = prior

    now = datetime.now(timezone.utc)
    return EntityEdge(
        uuid=uuid_,
        group_id="test",
        source_node_uuid="00000000-0000-0000-0000-000000000001",
        target_node_uuid="00000000-0000-0000-0000-000000000002",
        name="ASSERTS",
        fact=fact,
        created_at=now,
        expired_at=now if expired else None,
        attributes=attrs,
    )


@pytest.fixture(autouse=True)
def _need_graphiti():
    pytest.importorskip("graphiti_core")


class TestPromotedEdges:
    def test_belief_edge_becomes_confidence_change(self):
        kref = "kref://Atlas/Beliefs/zenith_floor.belief"
        edge = _edge("e1", kref=kref, confidence=0.2, prior=0.9, fact="Floor is $49.")

        changes = episode_edges_to_changes([edge], [])

        assert changes == [
            BeliefConfidenceChange(
                upstream_kref=kref,
                old_confidence=0.9,
                new_confidence=0.2,
                belief_text="Floor is $49.",
            )
        ]

    def test_missing_prior_defaults_to_new_confidence(self):
        kref = "kref://Atlas/Beliefs/x.belief"
        edge = _edge("e1", kref=kref, confidence=0.7)

        [change] = episode_edges_to_changes([edge], [])

        # No prior recorded -> no-op delta (old == new), cascade still safe.
        assert change.old_confidence == 0.7
        assert change.new_confidence == 0.7

    def test_missing_confidence_defaults_to_promoted_trust(self):
        kref = "kref://Atlas/Beliefs/x.belief"
        edge = _edge("e1", kref=kref)

        [change] = episode_edges_to_changes([edge], [])

        # Promotion == trust 1.0.
        assert change.new_confidence == 1.0
        assert change.old_confidence == 1.0

    def test_edge_without_kref_is_skipped(self):
        edge = _edge("e1", confidence=0.5)  # structural / non-belief edge
        assert episode_edges_to_changes([edge], []) == []

    def test_edge_with_unparseable_kref_is_skipped(self):
        edge = _edge("e1", kref="not-a-kref", confidence=0.5)
        assert episode_edges_to_changes([edge], []) == []

    def test_non_numeric_confidence_falls_back_to_default(self):
        kref = "kref://Atlas/Beliefs/x.belief"
        edge = _edge("e1", kref=kref)
        edge.attributes["confidence_score"] = "high"  # non-numeric

        [change] = episode_edges_to_changes([edge], [])
        assert change.new_confidence == 1.0

    def test_duplicate_krefs_collapse_last_wins(self):
        kref = "kref://Atlas/Beliefs/dup.belief"
        e1 = _edge("e1", kref=kref, confidence=0.5, prior=0.9)
        e2 = _edge("e2", kref=kref, confidence=0.3, prior=0.9)

        changes = episode_edges_to_changes([e1, e2], [])
        assert len(changes) == 1
        assert changes[0].new_confidence == 0.3


class TestInvalidatedEdges:
    def test_invalidated_belief_collapses_to_zero(self):
        kref = "kref://Atlas/Beliefs/retracted.belief"
        edge = _edge("e1", kref=kref, confidence=0.8, expired=True)

        [change] = episode_edges_to_changes([], [edge])

        assert change.upstream_kref == kref
        assert change.old_confidence == 0.8
        assert change.new_confidence == 0.0

    def test_retraction_overrides_promotion_for_same_kref(self):
        kref = "kref://Atlas/Beliefs/both.belief"
        promoted = _edge("e1", kref=kref, confidence=0.6, prior=0.9)
        invalidated = _edge("e2", kref=kref, confidence=0.6, prior=0.9, expired=True)

        changes = episode_edges_to_changes([promoted], [invalidated])

        assert len(changes) == 1
        assert changes[0].new_confidence == 0.0


class TestOrdering:
    def test_stable_first_seen_order(self):
        k1 = "kref://Atlas/Beliefs/a.belief"
        k2 = "kref://Atlas/Beliefs/b.belief"
        e1 = _edge("e1", kref=k1, confidence=0.5)
        e2 = _edge("e2", kref=k2, confidence=0.5)

        changes = episode_edges_to_changes([e1, e2], [])
        assert [c.upstream_kref for c in changes] == [k1, k2]

    def test_empty_inputs_yield_no_changes(self):
        assert episode_edges_to_changes([], []) == []
