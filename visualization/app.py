"""
Streamlit visualization app for the Layer10 Memory Graph.
Interactive graph explorer with evidence panel and merge audit.
"""

import json
import streamlit as st
from pathlib import Path
from pyvis.network import Network
import tempfile

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from graph.store import MemoryStore
from graph.memory_graph import MemoryGraph
from retrieval.retriever import Retriever

st.set_page_config(
    page_title="Layer10 Memory Graph",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_store():
    return MemoryStore()

@st.cache_resource
def load_graph():
    store = load_store()
    graph = MemoryGraph(store)
    graph.build_from_store()
    return graph

@st.cache_resource
def load_retriever():
    store = load_store()
    graph = load_graph()
    retriever = Retriever(store, graph)
    if not retriever.load_index():
        retriever.build_index()
    return retriever

st.sidebar.title("🧠 Layer10 Memory Graph")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🕸️ Graph Explorer",
    "🔍 Question Retrieval",
    "📋 Entity Browser",
    "📝 Claim Browser",
    "🔀 Merge Audit",
])

ENTITY_COLORS = {
    "person": "#4CAF50",
    "component": "#2196F3",
    "feature": "#FF9800",
    "bug": "#F44336",
    "release": "#9C27B0",
    "label": "#607D8B",
    "repository": "#795548",
    "team": "#00BCD4",
}


def render_graph_pyvis(entities: list[dict], edges: list[dict],
                        height: str = "600px") -> str:
    """Render a graph visualization using PyVis."""
    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
    )
    net.barnes_hut(gravity=-30000, central_gravity=0.3, spring_length=200)

    added_nodes = set()
    for e in entities:
        eid = e.get("id", "")
        if eid in added_nodes:
            continue
        added_nodes.add(eid)
        etype = e.get("entity_type", "unknown")
        color = ENTITY_COLORS.get(etype, "#999999")
        label = e.get("name", eid)
        title = f"Type: {etype}\nAliases: {', '.join(e.get('aliases', []))}"
        size = 20 + min(len(e.get("aliases", [])) * 3, 20)
        net.add_node(eid, label=label, title=title, color=color, size=size)

    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src in added_nodes and tgt in added_nodes:
            rel = edge.get("relation_type") or edge.get("claim_type", "related")
            conf = edge.get("confidence", "medium")
            width = {"high": 3, "medium": 2, "low": 1}.get(conf, 2)
            color = "#666666"
            if edge.get("temporal_status") == "historical":
                color = "#444444"
            net.add_edge(src, tgt, label=rel, title=edge.get("content", ""),
                        width=width, color=color)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        net.save_graph(f.name)
        return f.name


if page == "📊 Dashboard":
    st.title("📊 Memory Graph Dashboard")
    store = load_store()
    stats = store.get_stats()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Entities", stats.get("entities", 0))
    col2.metric("Claims", stats.get("claims", 0))
    col3.metric("Evidence", stats.get("evidence", 0))
    col4.metric("Merges", stats.get("merges", 0))
    col5.metric("Ingestions", stats.get("ingestions", 0))

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Entities by Type")
        etype_data = stats.get("entities_by_type", {})
        if etype_data:
            import pandas as pd
            df = pd.DataFrame(list(etype_data.items()), columns=["Type", "Count"])
            st.bar_chart(df.set_index("Type"))
        else:
            st.info("No entities yet. Run the pipeline first.")

    with col_b:
        st.subheader("Claims by Type")
        ctype_data = stats.get("claims_by_type", {})
        if ctype_data:
            import pandas as pd
            df = pd.DataFrame(list(ctype_data.items()), columns=["Type", "Count"])
            st.bar_chart(df.set_index("Type"))
        else:
            st.info("No claims yet. Run the pipeline first.")

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Claims by Confidence")
        conf_data = stats.get("claims_by_confidence", {})
        if conf_data:
            for k, v in conf_data.items():
                st.write(f"  **{k}**: {v}")

    with col_d:
        st.subheader("Claims by Temporal Status")
        ts_data = stats.get("claims_by_status", {})
        if ts_data:
            for k, v in ts_data.items():
                st.write(f"  **{k}**: {v}")

    graph = load_graph()
    summary = graph.get_graph_summary()
    st.subheader("Graph Topology")
    st.write(f"**Nodes:** {summary['num_nodes']} | **Edges:** {summary['num_edges']} | "
             f"**Components:** {summary['connected_components']} | "
             f"**Density:** {summary['density']:.4f}")

    if summary.get("top_degree_nodes"):
        st.subheader("Most Connected Entities")
        import pandas as pd
        df = pd.DataFrame(summary["top_degree_nodes"])
        st.dataframe(df, use_container_width=True)


elif page == "🕸️ Graph Explorer":
    st.title("🕸️ Graph Explorer")
    store = load_store()
    graph = load_graph()

    col1, col2, col3 = st.columns(3)
    with col1:
        entity_types = st.multiselect(
            "Filter by entity type",
            options=["person", "component", "feature", "bug", "release", "label", "repository", "team"],
            default=["person", "component", "feature"],
        )
    with col2:
        confidence_filter = st.multiselect(
            "Filter by confidence",
            options=["high", "medium", "low"],
            default=["high", "medium"],
        )
    with col3:
        temporal_filter = st.multiselect(
            "Filter by temporal status",
            options=["current", "historical", "disputed"],
            default=["current"],
        )

    all_entities = store.get_all_entities()
    entity_names = {e["id"]: e["name"] for e in all_entities}
    focus_entity = st.selectbox(
        "Focus on entity (optional)",
        options=["(All)"] + [f"{e['name']} ({e['entity_type']})" for e in all_entities[:100]],
    )

    if focus_entity != "(All)":
        focus_name = focus_entity.split(" (")[0]
        focus_id = None
        for e in all_entities:
            if e["name"] == focus_name:
                focus_id = e["id"]
                break
        if focus_id:
            neighborhood = graph.get_entity_neighborhood(focus_id, depth=2)
            display_entities = [neighborhood["entity"]] + neighborhood["neighbors"]
            display_edges = neighborhood["edges"]
        else:
            display_entities = []
            display_edges = []
    else:
        display_entities = [e for e in all_entities if e.get("entity_type") in entity_types]
        all_claims = store.get_all_claims()
        display_edges = []
        for c in all_claims:
            if c.get("object_entity_id"):
                if c.get("confidence", "medium") in confidence_filter:
                    if c.get("temporal_status", "current") in temporal_filter:
                        display_edges.append({
                            "source": c["subject_entity_id"],
                            "target": c["object_entity_id"],
                            "claim_id": c["id"],
                            "relation_type": c.get("relation_type"),
                            "claim_type": c.get("claim_type"),
                            "content": c.get("content"),
                            "confidence": c.get("confidence"),
                            "temporal_status": c.get("temporal_status"),
                        })

    if display_entities:
        html_path = render_graph_pyvis(display_entities, display_edges)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=620, scrolling=True)
        st.caption(f"Showing {len(display_entities)} entities and {len(display_edges)} edges")
    else:
        st.info("No entities match the current filters. Run the pipeline first or adjust filters.")


elif page == "🔍 Question Retrieval":
    st.title("🔍 Question Retrieval")
    st.markdown("Ask a question to retrieve grounded context from the memory graph.")

    question = st.text_input("Your question:", placeholder="e.g. What is React Suspense and who worked on it?")
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Max results", 3, 30, 10)
    with col2:
        include_hist = st.checkbox("Include historical claims", value=True)

    if st.button("🔍 Retrieve", type="primary") and question:
        retriever = load_retriever()
        with st.spinner("Retrieving..."):
            pack = retriever.retrieve(question, top_k=top_k, include_historical=include_hist)

        st.success(pack.summary)

        if pack.entities:
            st.subheader("Relevant Entities")
            for e in pack.entities:
                with st.expander(f"🏷️ {e.get('name')} ({e.get('entity_type')})"):
                    st.write(f"**ID:** {e.get('id')}")
                    if e.get("aliases"):
                        st.write(f"**Aliases:** {', '.join(e['aliases'])}")
                    if e.get("properties"):
                        st.json(e["properties"])

        if pack.claims:
            st.subheader("Relevant Claims")
            for i, c in enumerate(pack.claims, 1):
                status_icon = {"current": "🟢", "historical": "🟡", "disputed": "🔴"}.get(
                    c.get("temporal_status", "current"), "⚪"
                )
                conf_icon = {"high": "⬆️", "medium": "↔️", "low": "⬇️"}.get(
                    c.get("confidence", "medium"), "↔️"
                )
                with st.expander(f"{status_icon} {conf_icon} [{c.get('claim_type')}] {c.get('content', '')[:100]}"):
                    st.write(f"**Full claim:** {c.get('content')}")
                    st.write(f"**Subject:** {c.get('subject_entity_id')}")
                    if c.get("object_entity_id"):
                        st.write(f"**Object:** {c['object_entity_id']}")
                    st.write(f"**Confidence:** {c.get('confidence')} | **Status:** {c.get('temporal_status')}")
                    if c.get("valid_from"):
                        st.write(f"**Valid from:** {c['valid_from']}")
                    if c.get("valid_until"):
                        st.write(f"**Valid until:** {c['valid_until']}")

                    st.markdown("---")
                    st.markdown("**📎 Supporting Evidence:**")
                    for ev in c.get("evidence", []):
                        st.info(f"📄 **Source:** {ev.get('source_id', 'unknown')}\n\n"
                               f"> {ev.get('excerpt', 'N/A')}\n\n"
                               f"🔗 [{ev.get('url', '')}]({ev.get('url', '')})")

        if pack.conflicts:
            st.subheader("⚠️ Conflicting Information")
            for conf in pack.conflicts:
                st.warning(f"**{conf.get('claim_type')}** about `{conf.get('subject')}`:\n\n"
                          f"🟢 Current: {conf.get('current')}\n\n"
                          f"🟡 Historical: {conf.get('historical')}")

        with st.expander("📋 Raw Context Pack (JSON)"):
            st.json(pack.to_dict())


elif page == "📋 Entity Browser":
    st.title("📋 Entity Browser")
    store = load_store()

    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search entities:", placeholder="e.g. dan, suspense, fiber")
    with col2:
        type_filter = st.selectbox("Filter by type", ["All"] +
            ["person", "component", "feature", "bug", "release", "label", "repository", "team"])

    etype = type_filter if type_filter != "All" else None
    if search_term:
        entities = store.search_entities(search_term, etype)
    else:
        entities = store.get_all_entities()
        if etype:
            entities = [e for e in entities if e["entity_type"] == etype]

    st.write(f"Found **{len(entities)}** entities")

    for e in entities[:50]:
        with st.expander(f"🏷️ {e['name']} ({e['entity_type']}) — {e['id']}"):
            st.write(f"**ID:** {e['id']}")
            st.write(f"**Type:** {e['entity_type']}")
            if e.get("aliases"):
                st.write(f"**Aliases:** {', '.join(e['aliases'])}")
            if e.get("first_seen"):
                st.write(f"**First seen:** {e['first_seen']}")
            if e.get("last_seen"):
                st.write(f"**Last seen:** {e['last_seen']}")
            if e.get("properties"):
                st.json(e["properties"])

            claims = store.get_claims_for_entity(e["id"])
            if claims:
                st.markdown("**Claims:**")
                for c in claims[:10]:
                    status = c.get("temporal_status", "current")
                    icon = {"current": "🟢", "historical": "🟡", "disputed": "🔴"}.get(status, "⚪")
                    st.write(f"  {icon} [{c.get('claim_type')}] {c.get('content', '')[:150]}")
                    for ev in c.get("evidence", [])[:2]:
                        st.caption(f"    📎 {ev.get('excerpt', '')[:100]}...")

            if e.get("merge_history"):
                st.markdown("**Merge History:**")
                for m in e["merge_history"]:
                    st.write(f"  🔀 Merged `{m.get('merged_name', '?')}` via {m.get('method', '?')}")


elif page == "📝 Claim Browser":
    st.title("📝 Claim Browser")
    store = load_store()

    col1, col2, col3 = st.columns(3)
    with col1:
        search_term = st.text_input("Search claims:", placeholder="e.g. suspense, concurrent")
    with col2:
        claim_type_filter = st.selectbox("Claim type", ["All"] +
            ["decision", "status_change", "assignment", "bug_report", "feature_request",
             "technical_fact", "dependency", "proposal", "agreement", "disagreement",
             "workaround", "root_cause", "resolution"])
    with col3:
        status_filter = st.selectbox("Temporal status", ["All", "current", "historical", "disputed"])

    ctype = claim_type_filter if claim_type_filter != "All" else None
    tstatus = status_filter if status_filter != "All" else None

    if search_term:
        claims = store.search_claims(search_term, ctype, tstatus)
    else:
        claims = store.get_all_claims()
        if ctype:
            claims = [c for c in claims if c["claim_type"] == ctype]
        if tstatus:
            claims = [c for c in claims if c.get("temporal_status") == tstatus]

    st.write(f"Found **{len(claims)}** claims")

    for c in claims[:50]:
        status = c.get("temporal_status", "current")
        conf = c.get("confidence", "medium")
        icon = {"current": "🟢", "historical": "🟡", "disputed": "🔴"}.get(status, "⚪")

        with st.expander(f"{icon} [{c.get('claim_type')}] {c.get('content', '')[:120]}"):
            st.write(f"**Claim:** {c.get('content')}")
            st.write(f"**Type:** {c.get('claim_type')} | **Confidence:** {conf} | **Status:** {status}")
            st.write(f"**Subject:** {c.get('subject_entity_id')}")
            if c.get("object_entity_id"):
                st.write(f"**Object:** {c['object_entity_id']}")
            if c.get("valid_from"):
                st.write(f"**Valid from:** {c['valid_from']}")
            if c.get("valid_until"):
                st.write(f"**Valid until:** {c['valid_until']}")
            if c.get("superseded_by"):
                st.write(f"**Superseded by:** {c['superseded_by']}")
            if c.get("merged_from"):
                st.write(f"**Merged from:** {', '.join(c['merged_from'])}")

            st.markdown("---")
            st.markdown("**📎 Evidence:**")
            for ev in c.get("evidence", []):
                st.info(f"📄 **Source:** {ev.get('source_id', 'unknown')} ({ev.get('source_type', '?')})\n\n"
                       f"> {ev.get('excerpt', 'N/A')}\n\n"
                       f"🔗 [{ev.get('url', '')}]({ev.get('url', '')})")


elif page == "🔀 Merge Audit":
    st.title("🔀 Merge & Deduplication Audit")
    store = load_store()

    merges = store.get_merge_log()
    st.write(f"Total merge operations: **{len(merges)}**")

    if merges:
        by_action = {}
        for m in merges:
            action = m.get("action", "unknown")
            by_action.setdefault(action, []).append(m)

        for action, items in by_action.items():
            st.subheader(f"{action} ({len(items)})")
            for m in items[:20]:
                with st.expander(f"🔀 {m.get('canonical_id', '?')} ← {m.get('merged_id', '?')}"):
                    st.write(f"**Method:** {m.get('method', '?')}")
                    st.write(f"**Reason:** {m.get('reason', 'N/A')}")
                    st.write(f"**Time:** {m.get('created_at', 'N/A')}")
                    if m.get("details"):
                        try:
                            details = json.loads(m["details"]) if isinstance(m["details"], str) else m["details"]
                            st.json(details)
                        except Exception:
                            st.write(str(m["details"]))
    else:
        st.info("No merges recorded yet. Run the pipeline first.")
