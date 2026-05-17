# agent/tools.py

import arxiv
import json
import os
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list
    url: str
    published: str


def fetch_papers(topic: str, max_results: int = 10) -> list:
    """Fetches latest papers from arXiv for a given topic."""

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in search.results():
        paper = Paper(
            id=result.entry_id,
            title=result.title.strip(),
            abstract=result.summary,
            authors=[a.name for a in result.authors],
            url=result.pdf_url,
            published=str(result.published.date())
        )
        papers.append(paper)

    return papers


def summarise_paper(paper: Paper) -> dict:
    """Uses Gemini 2.0 Flash (free) to summarise a paper."""

    SUMMARISE_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a research assistant helping an early-career ML engineer
stay up to date with AI research papers.

Given a paper title and abstract, return a JSON object with exactly these fields:
{{
  "summary": "2 sentences in plain English. No jargon. What did they do and what did they find?",
  "contribution": "The single most important thing this paper adds to the field. One sentence.",
  "relevance": "Why should a fresher ML engineer building AI applications care about this? One sentence.",
  "difficulty": "beginner or intermediate or advanced"
}}

Rules:
- Return ONLY valid JSON. No markdown, no explanation, no preamble.
- Use simple language a computer science undergraduate can understand."""
        ),
        (
            "human",
            "Title: {title}\n\nAbstract: {abstract}"
        )
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    chain = SUMMARISE_PROMPT | llm

    try:
        response = chain.invoke({
            "title": paper.title,
            "abstract": paper.abstract
        })

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        result = json.loads(content)
        return result

    except json.JSONDecodeError:
        return {
            "summary": paper.abstract[:300] + "...",
            "contribution": "Could not parse automatically.",
            "relevance": "Read the abstract directly.",
            "difficulty": "unknown"
        }