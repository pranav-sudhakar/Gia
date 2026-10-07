from tavily import TavilyClient

tavily_client = TavilyClient(api_key="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")

MAX_CHARS_PER_SOURCE = 1500

def web_search(query: str) -> str:
    """Search the web for a query and return short snippets from the top
    results, each with a URL and a relevance score.

    Use this for general web information — current events, non-Wikipedia
    sources, or anything not well-suited to an encyclopedia entry. For
    questions about a specific well-known person, place, organization, or
    historical fact, try wikipedia_search first — it's faster and usually
    gives a cleaner, more direct answer for that category of question.
    Fall back to web_search when wikipedia_search doesn't have a relevant
    article, or when the question needs broader web coverage.

    The snippets returned here are short and may not contain the full
    detail needed to answer the question. Judge each result's relevance
    using its score and content — if none of the snippets are detailed
    enough, follow up with web_extract on the single most promising URL.
    Do not call web_extract on a URL unless the snippet from this search
    genuinely looked relevant; extracting is more expensive, so it should
    only be used when needed, not on every search.
    """
    try:
        response = tavily_client.search(query, max_results=5)
        results = response["results"]

        if not results:
            return "No search results found."

        formatted = []
        for r in results:
            formatted.append(f"URL: {r['url']}\nScore: {r['score']:.2f}\nSnippet: {r['content']}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search failed due to an error: {e}. Try a different query or tool."


def web_extract(url: str, query: str) -> str:
    """Extract the most relevant content from a specific URL, filtered to
    what's relevant to your query. Use this only after web_search (or
    wikipedia_search), and only when the snippet you already have wasn't
    detailed enough to answer the question — this is a last resort for
    depth, not a first step.

    Pass the exact URL as returned by web_search, and a specific query
    describing what information you're looking for on that page (e.g.
    "the release date of this film", not just the original question
    verbatim) — the more specific the query, the more relevant the
    returned content will be.

    If the extracted content still doesn't answer the question, do not
    extract further URLs speculatively — instead go back to web_search
    with a reworded query, or reconsider whether a different tool (e.g.
    python_repl for a calculation, or wikipedia_search for a related
    entity) is a better fit for what's actually being asked.
    """
    try:
        response = tavily_client.extract(
            urls=[url],
            query=query,
            chunks_per_source=2,
        )
        results = response["results"]

        if not results:
            return f"Could not extract content from {url}."

        content = results[0]["raw_content"][:MAX_CHARS_PER_SOURCE]
        return f"Source: {url}\n{content}"
    except Exception as e:
        return f"Search failed due to an error: {e}. Try a different query or tool."

