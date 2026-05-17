# app.py

import streamlit as st
from agent.graph import agent
from agent.memory import PaperMemory

st.set_page_config(page_title="Research Agent", page_icon="🤖", layout="wide")

st.title("🤖 Autonomous Research Agent")
st.caption("Tracks arXiv papers weekly so you stay current on AI research")

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    available_topics = [
        "retrieval augmented generation",
        "large language model agents",
        "efficient llm inference",
        "multimodal language models",
        "AI safety alignment",
        "reinforcement learning from human feedback",
        "vision transformers",
        "graph neural networks",
        "diffusion models",
        "federated learning"
    ]

    selected_topics = st.multiselect(
        "Research topics to track",
        available_topics,
        default=["retrieval augmented generation", "large language model agents"]
    )

    st.divider()

    try:
        memory = PaperMemory()
        st.metric("Papers in memory", memory.count())
        st.caption("Total papers seen across all runs")
    except Exception:
        st.metric("Papers in memory", "—")

# ── Main area ──────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Run Agent")

    if not selected_topics:
        st.warning("Select at least one topic in the sidebar.")
    else:
        if st.button("Fetch & Summarise New Papers", type="primary"):
            with st.status("Agent running...", expanded=True) as status:
                st.write(f"Searching {len(selected_topics)} topics on arXiv...")

                initial_state = {
                    "topics": selected_topics,
                    "fetched_papers": [],
                    "new_papers": [],
                    "summaries": [],
                    "digest": "",
                    "approved": True,
                    "error": "",
                    "messages": []
                }

                result = agent.invoke(initial_state)
                summaries = result.get("summaries", [])
                st.write(f"Found {len(summaries)} new papers.")
                status.update(label="Done!", state="complete")

            if summaries:
                st.subheader(f"{len(summaries)} New Papers This Week")

                for i, s in enumerate(summaries):
                    with st.expander(f"{s['title'][:80]}...", expanded=(i == 0)):
                        st.caption(f"Published: {s['published']} | Difficulty: {s.get('difficulty', '?')}")
                        st.markdown(f"**Summary:** {s.get('summary', '—')}")
                        st.markdown(f"**Key contribution:** {s.get('contribution', '—')}")
                        st.markdown(f"**Why you should care:** {s.get('relevance', '—')}")
                        st.markdown(f"[Read full paper]({s['url']})")

                st.download_button(
                    "Download digest as Markdown",
                    data=result.get("digest", ""),
                    file_name="weekly_digest.md",
                    mime="text/markdown"
                )
            else:
                st.info("No new papers found. All recent papers already summarised.")

with col2:
    st.subheader("Search Memory")
    st.caption("Find past papers by meaning, not keywords")

    query = st.text_input("Search query", placeholder="e.g. faster inference for LLMs")

    if query:
        try:
            memory = PaperMemory()
            results = memory.search_similar(query, n_results=5)

            if results:
                for r in results:
                    with st.container():
                        st.markdown(f"**{r['title']}**")
                        st.caption(f"{r.get('published', '?')} — {r.get('authors', '?')}")
                        st.markdown(f"[Read →]({r['url']})")
                        st.divider()
            else:
                st.info("No papers in memory yet. Run the agent first.")
        except Exception as e:
            st.error(f"Search error: {e}")