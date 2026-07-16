"""End-to-end: AtlasGraphiti.add_episode -> Ripple cascade against live Neo4j.

Regression coverage for A3: the flagship ingestion hook invoked
`RippleEngine.propagate(new_edges=..., invalidated_edges=..., episode=...)` — a
signature the engine never exposed — so the composition raised `TypeError` the
instant `ripple_engine` + `ledger` were wired. The only prior coverage was an
instantiation smoke test that left both stubs `None`.

These tests wire a real `RippleEngine` + a stub ledger into `AtlasGraphiti`,
feed a canned `AddEpisodeResults` (bypassing the LLM extractor), and prove the
cascade actually runs against live Neo4j.

Spec: notes/06 - Ripple Algorithm Spec.md § 4
"""

import os
import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def neo4j_uri() -> str:
    return os.environ.get("NEO4J_URI", "bolt://localhost:7687")


@pytest.fixture(scope="module")
def neo4j_auth() -> tuple[str, str]:
    return (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "atlasdev"),
    )


@pytest.fixture
async def driver(neo4j_uri, neo4j_auth):
    pytest.importorskip("neo4j")
    from neo4j import AsyncGraphDatabase

    user, password = neo4j_auth
    drv = AsyncGraphDatabase.driver(neo4j_uri, auth=(user, password))
    try:
        await drv.verify_connectivity()
        yield drv
    finally:
        await drv.close()


@pytest.fixture
def ns() -> str:
    return f"AddEpRipple_{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def cleanup(driver, ns):
    cypher = "MATCH (n) WHERE n.kref STARTS WITH $p DETACH DELETE n"
    prefix = f"kref://{ns}/"
    async with driver.session() as s:
        await s.run(cypher, p=prefix)
    yield
    async with driver.session() as s:
        await s.run(cypher, p=prefix)


class _StubLedger:
    """Reports the given edge uuids as promoted to the trust ledger."""

    def __init__(self, promoted: set[str]):
        self._promoted = promoted

    def is_promoted(self, edge_uuid: str) -> bool:
        return edge_uuid in self._promoted


class _RecordingEngine:
    """Wraps a real RippleEngine so the test can inspect what the hook cascaded
    without re-walking the graph. Delegates to the real engine end-to-end."""

    def __init__(self, engine):
        self._engine = engine
        self.cascades = []

    async def propagate(self, *args, **kwargs):
        result = await self._engine.propagate(*args, **kwargs)
        self.cascades.append(result)
        return result


def _entity_edge(uuid_: str, *, kref: str, confidence: float, prior: float,
                 fact: str, expired: bool = False):
    from graphiti_core.edges import EntityEdge

    now = datetime.now(timezone.utc)
    return EntityEdge(
        uuid=uuid_,
        group_id="test",
        source_node_uuid=str(uuid.uuid4()),
        target_node_uuid=str(uuid.uuid4()),
        name="ASSERTS",
        fact=fact,
        created_at=now,
        expired_at=now if expired else None,
        attributes={
            "kref": kref,
            "confidence_score": confidence,
            "prior_confidence": prior,
        },
    )


def _canned_results(edges):
    from graphiti_core.graphiti import AddEpisodeResults
    from graphiti_core.nodes import EpisodicNode

    episode = EpisodicNode(
        name="ep",
        group_id="test",
        labels=[],
        source="text",
        source_description="add_episode ripple test",
        content="An upstream belief changed.",
        valid_at=datetime.now(timezone.utc),
        entity_edges=[],
    )
    return AddEpisodeResults(
        episode=episode,
        episodic_edges=[],
        nodes=[],
        edges=edges,
        communities=[],
        community_edges=[],
    )


async def _wire_atlas(driver, ledger, engine):
    """Build an AtlasGraphiti without the full Graphiti/LLM stack — we only
    exercise add_episode + its ripple/ledger attributes."""
    from atlas_core.graphiti import AtlasGraphiti

    atlas = AtlasGraphiti.__new__(AtlasGraphiti)
    atlas.ripple_engine = engine
    atlas.ledger = ledger
    atlas.quarantine_store = None
    return atlas


class TestAddEpisodeCascade:
    async def test_promoted_belief_edge_runs_cascade(self, driver, ns, monkeypatch):
        """add_episode on a promoted belief edge cascades to its DEPENDS_ON
        dependent — and does NOT raise (the A3 regression)."""
        import atlas_core.graphiti as gmod
        from atlas_core.ripple import RippleEngine

        upstream = f"kref://{ns}/Beliefs/upstream.belief"
        downstream = f"kref://{ns}/Beliefs/downstream.belief"
        async with driver.session() as s:
            await s.run(
                "MERGE (a:AtlasItem {kref: $a}) SET a.deprecated = false, "
                "  a.confidence_score = 0.2 "
                "MERGE (b:AtlasItem {kref: $b}) SET b.deprecated = false, "
                "  b.confidence_score = 0.85, b.last_evidence_days = 0 "
                "MERGE (b)-[:DEPENDS_ON {strength: 0.8}]->(a)",
                a=upstream, b=downstream,
            )

        edge_uuid = str(uuid.uuid4())
        edge = _entity_edge(
            edge_uuid, kref=upstream, confidence=0.2, prior=0.9,
            fact="Zenith pricing floor dropped.",
        )
        results = _canned_results([edge])

        engine = _RecordingEngine(RippleEngine(driver, emit_events=False))
        atlas = await _wire_atlas(driver, _StubLedger({edge_uuid}), engine)

        async def _fake_super(self, *args, **kwargs):
            return results

        monkeypatch.setattr(gmod.Graphiti, "add_episode", _fake_super)

        returned = await atlas.add_episode()

        # add_episode returns the ingestion results unchanged.
        assert returned is results
        # Exactly one cascade fired, for the upstream belief.
        assert len(engine.cascades) == 1
        cascade = engine.cascades[0]
        assert cascade.succeeded is True
        assert cascade.origin_kref == upstream
        # The downstream dependent was impacted and a proposal produced.
        assert cascade.n_impacted == 1
        assert {p.target_kref for p in cascade.proposals} == {downstream}

    async def test_no_promoted_edges_skips_cascade(self, driver, ns, monkeypatch):
        """If no edge is promoted, the hook never touches the engine."""
        import atlas_core.graphiti as gmod
        from atlas_core.ripple import RippleEngine

        edge = _entity_edge(
            str(uuid.uuid4()),
            kref=f"kref://{ns}/Beliefs/unpromoted.belief",
            confidence=0.5, prior=0.9, fact="Not promoted.",
        )
        results = _canned_results([edge])

        engine = _RecordingEngine(RippleEngine(driver, emit_events=False))
        atlas = await _wire_atlas(driver, _StubLedger(set()), engine)  # nothing promoted

        async def _fake_super(self, *args, **kwargs):
            return results

        monkeypatch.setattr(gmod.Graphiti, "add_episode", _fake_super)

        await atlas.add_episode()
        assert engine.cascades == []

    async def test_engine_not_wired_is_noop(self, driver, ns, monkeypatch):
        """add_episode with ripple_engine=None just returns ingestion results."""
        import atlas_core.graphiti as gmod

        edge = _entity_edge(
            str(uuid.uuid4()),
            kref=f"kref://{ns}/Beliefs/x.belief",
            confidence=0.5, prior=0.9, fact="x",
        )
        results = _canned_results([edge])

        atlas = await _wire_atlas(driver, _StubLedger({edge.uuid}), engine=None)

        async def _fake_super(self, *args, **kwargs):
            return results

        monkeypatch.setattr(gmod.Graphiti, "add_episode", _fake_super)

        returned = await atlas.add_episode()
        assert returned is results
