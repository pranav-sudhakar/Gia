from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from multimodal_input import analyze_image, read_docx, read_pdf, read_spreadsheet, transcribe_audio
from youtube_trancripts import get_youtube_transcript
from tavily_api import web_extract, web_search
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from python_repl import python_repl
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1200)
wikipedia_search = WikipediaQueryRun(api_wrapper=wiki_wrapper)

def wikipedia_lookup(query: str) -> str:
    """Search Wikipedia for a query and return a summary. Use this first for
    questions about a specific well-known person, place, organization, or
    historical fact. Fall back to web_search if this doesn't have a
    relevant article or returns an error."""
    try:
        return wikipedia_search.run(query)
    except Exception as e:
        return f"Wikipedia lookup failed: {e}. Try web_search instead."

class AgentState(TypedDict):
    # The document provided
    file_path: Optional[str]  # Contains file path (PDF/PNG)
    messages: Annotated[list[AnyMessage], add_messages]

tools = [
    analyze_image,
    read_docx,
    read_pdf,
    read_spreadsheet,
    web_extract,
    web_search,
    python_repl,
    transcribe_audio,
    get_youtube_transcript,
    wikipedia_lookup
]

llm = ChatAnthropic(model="claude-sonnet-4-6")
bound_llm = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a general AI assistant answering questions using \
the tools available to you.

Reason step by step. Before answering, decide what information you need and \
which tool can get it. Do not answer from memory alone if a tool can verify \
the fact — questions are designed so that unverified guessing is usually wrong.

If a file is attached to this question, use the appropriate tool to read it \
before reasoning about its content: images need analyze_image, PDFs need \
read_pdf, spreadsheets need read_spreadsheet, Word documents need read_docx, \
audio needs transcribe_audio.

Once you have gathered enough information to answer confidently, stop calling \
tools and give your final answer.

Report your final answer as the very last line of your response, in exactly \
this format:

FINAL ANSWER: [YOUR ANSWER]

Formatting rules for YOUR ANSWER:
- Numbers: digits only, no commas, no units ($, %, km, etc.) unless the \
question explicitly asks for units.
- Strings: no articles (a/an/the), no abbreviations unless asked, digits \
written as digits not words.
- Lists: apply the above rules to each comma-separated element.
- Never leave this blank — if unsure, give your best guess in the correct \
format rather than an empty or explanatory answer.

If the question itself specifies an exact output format (e.g. "comma-separated
list", "alphabetized", "no measurements"), your final answer must follow that
format exactly — no introductory phrases, no labels, no extra words. Only the
requested content itself, in the requested order.

For any task requiring exact sorting, alphabetizing, or counting, use
python_repl to compute it programmatically (e.g. sorted(list)) rather than
ordering it yourself — manual ordering is error-prone.

If a tool returns the same type of error twice in a row, stop retrying that
tool and either try a different tool or give your best answer based on what
you have so far.
"""

def assistant(state: AgentState):
    input_file = state.get("file_path")
    file_note = (
        f"A file is attached to this question, located at: {input_file}"
        if input_file else ""
    )

    # cache_control is Anthropic-only — only attach it when actually running on Claude
    if isinstance(llm, ChatAnthropic):
        system_content = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
        ]
        if file_note:
            system_content.append({"type": "text", "text": file_note})
        sys_msg = SystemMessage(content=system_content)
    else:
        sys_msg = SystemMessage(content=SYSTEM_PROMPT + (f"\n\n{file_note}" if file_note else ""))

    return {
        "messages": [bound_llm.invoke([sys_msg] + state["messages"])],
        "file_path": input_file,
    }

builder = StateGraph(AgentState)
builder.add_node("assistant",assistant)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant",tools_condition)
builder.add_edge("tools","assistant")
#while tracking the flow of this, you'll see that there's no END here, that's because add_conditional_edges automatically returns end if tools_condition doens't become true. if it does become true, it simply goes to the next line where there's aldready add_edge("tools","assistant")

react_graph = builder.compile() 