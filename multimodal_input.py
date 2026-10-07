import base64
import os
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")

def analyze_image(image_path: str, question: str) -> str:
    """Look at an image file and answer a specific question about it using
    a vision-capable model. Use this whenever the question references an
    attached image — photos, screenshots, charts, diagrams, scanned pages,
    or any visual content that needs to be interpreted rather than just
    read as text.

    Always pass a specific, focused question about what you need from the
    image (e.g. "What is the total shown in this chart?" rather than just
    "describe this image") — a targeted question gets a more precise and
    useful answer than a generic description.

    This makes a separate model call specifically to interpret the image,
    so only use it when the question actually requires visual understanding
    — do not call this speculatively on files that turn out to be text-based
    (use read_pdf, read_docx, or read_spreadsheet for those instead).
    """
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return (
            "Error: this looks like a URL, not a local file path. If this is a "
            "YouTube link, use get_youtube_transcript instead. This tool only "
            "works on local file paths from an attached file."
        )
    if not os.path.exists(image_path):
        return f"Error: file not found at '{image_path}'."
    
    with open(image_path, "rb") as f:
        image_base64 = base64.standard_b64encode(f.read()).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
    ])
    response = llm.invoke([message])
    return response.content

from faster_whisper import WhisperModel

whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to plain text. Use this whenever the
    question references an attached audio file (a recording, a voice
    memo, spoken instructions, etc.) and you need to know what was said
    in it.

    Returns the full transcribed text with no timestamps. If the audio
    is unclear, in a strong accent, or contains significant background
    noise, the transcription may contain minor errors — read the output
    critically rather than trusting it word-for-word, especially for
    names, numbers, or technical terms, and cross-check against other
    tools (e.g. web_search) if the answer hinges on a detail that looks
    uncertain.
    """
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        return (
            "Error: this looks like a URL, not a local file path. If this is a "
            "YouTube link, use get_youtube_transcript instead. This tool only "
            "works on local file paths from an attached file."
        )
    if not os.path.exists(audio_path):
        # try the common mismatch: model dropped the "downloaded_" prefix
        alt_path = f"downloaded_{audio_path}"
        if os.path.exists(alt_path):
            audio_path = alt_path
        else:
            return f"Error: file not found at '{audio_path}'. Check the exact path given in the system prompt."

    try:
        segments, _ = whisper_model.transcribe(audio_path)
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        return f"Error transcribing audio: {e}"

import pdfplumber
import pandas as pd
from docx import Document

def read_pdf(file_path: str) -> str:
    """Extract text content from a PDF file. Use this as the first step
    whenever the question has an attached PDF.

    Returns only the extractable text layer of the document, truncated to
    the first several thousand characters — sufficient for most text-based
    PDFs, but if the answer isn't found in the returned text, the relevant
    content may be further into the document or in a scanned/image page
    with no text layer.

    If this returns empty or clearly insufficient content (e.g. the PDF is
    a scanned image with no real text layer, or contains a chart/diagram
    whose content matters), do not guess — instead use analyze_image on
    the same file to visually interpret it.
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return (
            "Error: this looks like a URL, not a local file path. If this is a "
            "YouTube link, use get_youtube_transcript instead. This tool only "
            "works on local file paths from an attached file."
        )
    if not os.path.exists(file_path):
        return f"Error: file not found at '{file_path}'."
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text[:5000] if text else "No extractable text found in this PDF."
    except Exception as e:
        return f"Error reading this PDF: {e}. The downloaded file may be corrupted or incomplete."

def read_spreadsheet(file_path: str) -> str:
    """Read a spreadsheet file (.xlsx or .csv) and return its contents as
    plain text. Use this whenever the question has an attached spreadsheet
    and you need to see what data it contains.

    This returns the full table as text, which works well for small to
    medium spreadsheets. If the spreadsheet is large, or the question
    requires actual computation over the data (sums, averages, filtering,
    sorting, counting rows matching a condition, etc.), do not attempt the
    calculation by reading the text yourself — instead use python_repl
    with pandas (`pd.read_csv` / `pd.read_excel` on this same file path)
    to compute the answer precisely.
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return (
            "Error: this looks like a URL, not a local file path. If this is a "
            "YouTube link, use get_youtube_transcript instead. This tool only "
            "works on local file paths from an attached file."
        )
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    return df.to_string()


def read_docx(file_path: str) -> str:
    """Extract text content from a Word document (.docx). Use this
    whenever the question has an attached Word document.

    Returns the full paragraph text of the document in reading order.
    Note: this does not extract text from tables, headers/footers, or
    embedded images within the document — if the answer might be in one
    of those and isn't found in the returned paragraph text, treat the
    result as incomplete rather than conclusive.
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return (
            "Error: this looks like a URL, not a local file path. If this is a "
            "YouTube link, use get_youtube_transcript instead. This tool only "
            "works on local file paths from an attached file."
        )
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)