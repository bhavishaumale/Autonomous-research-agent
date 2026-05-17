# agent/graph.py

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from agent.tools import fetch_papers, summarise_paper, Paper
from agent.memory import PaperMemory


class AgentState(TypedDict):
    topics: list
    fetched_papers: list
    new_papers: list
    summaries: list
    digest: str
    approved: bool
    error: str
    messages: Annotated[list, add_messages]


def fetch_node(state: AgentState) -> dict:
    print(f"[Fetch] Searching arXiv for {len(state['topics'])} topics...")
    all_papers = []
    for topic in state["topics"]:
        print(f"  → Fetching: '{topic}'")
        try:
            papers = fetch_papers(topic, max_results=8)
            all_papers.extend([vars(p) for p in papers])
            print(f"    Found {len(papers)} papers")
        except Exception as e:
            print(f"    Error fetching '{topic}': {e}")
    print(f"[Fetch] Total fetched: {len(all_papers)} papers")
    return {"fetched_papers": all_papers}


def filter_node(state: AgentState) -> dict:
    print(f"[Filter] Checking {len(state['fetched_papers'])} papers against memory...")
    memory = PaperMemory()
    new_papers = []
    for paper_dict in state["fetched_papers"]:
        if memory.is_new(paper_dict["id"]):
            new_papers.append(paper_dict)
    skipped = len(state["fetched_papers"]) - len(new_papers)
    print(f"[Filter] {len(new_papers)} new papers, {skipped} already seen")
    return {"new_papers": new_papers}


def summarise_node(state: AgentState) -> dict:
    papers_to_process = state["new_papers"][:6]
    print(f"[Summarise] Summarising {len(papers_to_process)} papers...")
    summaries = []
    for i, paper_dict in enumerate(papers_to_process):
        print(f"  → [{i+1}/{len(papers_to_process)}] {paper_dict['title'][:60]}...")
        try:
            paper = Paper(**paper_dict)
            summary = summarise_paper(paper)
            full_entry = {**paper_dict, **summary}
            summaries.append(full_entry)
            memory = PaperMemory()
            memory.store(paper)
        except Exception as e:
            print(f"    Error summarising paper: {e}")
            continue
    print(f"[Summarise] Done. {len(summaries)} papers summarised.")
    return {"summaries": summaries}


def compile_digest_node(state: AgentState) -> dict:
    print("[Compile] Building digest...")
    if not state["summaries"]:
        return {"digest": "No new papers found this week.", "approved": False}
    lines = []
    lines.append("# Weekly AI Research Digest")
    lines.append(f"*{len(state['summaries'])} new papers across {len(state['topics'])} topics*")
    lines.append("")
    for difficulty in ["beginner", "intermediate", "advanced", "unknown"]:
        papers_at_level = [s for s in state["summaries"] if s.get("difficulty") == difficulty]
        if not papers_at_level:
            continue
        lines.append(f"## {difficulty.title()} Level")
        lines.append("")
        for s in papers_at_level:
            lines.append(f"### {s['title']}")
            lines.append(f"*Published: {s['published']} | Authors: {', '.join(s['authors'][:2])}*")
            lines.append("")
            lines.append(f"**What they did:** {s.get('summary', '—')}")
            lines.append(f"**Key contribution:** {s.get('contribution', '—')}")
            lines.append(f"**Why you should care:** {s.get('relevance', '—')}")
            lines.append(f"[Read paper →]({s['url']})")
            lines.append("")
            lines.append("---")
            lines.append("")
    digest = "\n".join(lines)
    print(f"[Compile] Digest ready — {len(digest)} characters")
    return {"digest": digest, "approved": False}


def human_approval_node(state: AgentState) -> dict:
    print("\n" + "="*60)
    print("DIGEST PREVIEW")
    print("="*60)
    print(state["digest"])
    print("="*60 + "\n")
    answer = input("Approve and send this digest? (yes/no): ").strip().lower()
    return {"approved": answer == "yes"}


def should_send_email(state: AgentState) -> str:
    if state.get("approved"):
        print("[Router] Approved — sending email.")
        return "send_email"
    else:
        print("[Router] Not approved — stopping.")
        return END


def send_email_node(state: AgentState) -> dict:
    from digest.emailer import send_email_digest
    print("[Send] Sending email digest...")
    try:
        send_email_digest(state["digest"])
        print("[Send] Email sent successfully!")
    except Exception as e:
        print(f"[Send] Email failed: {e}")
        return {"error": str(e)}
    return {}


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("fetch", fetch_node)
    graph.add_node("filter", filter_node)
    graph.add_node("summarise", summarise_node)
    graph.add_node("compile", compile_digest_node)
    graph.add_node("approve", human_approval_node)
    graph.add_node("send_email", send_email_node)
    graph.set_entry_point("fetch")
    graph.add_edge("fetch", "filter")
    graph.add_edge("filter", "summarise")
    graph.add_edge("summarise", "compile")
    graph.add_edge("compile", "approve")
    graph.add_conditional_edges(
        "approve",
        should_send_email,
        {"send_email": "send_email", END: END}
    )
    graph.add_edge("send_email", END)
    return graph.compile()


agent = build_agent()