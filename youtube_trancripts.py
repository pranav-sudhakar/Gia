import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

ytt_api = YouTubeTranscriptApi()

def get_youtube_transcript(url_or_id: str) -> str:
    """Get the transcript/captions of a YouTube video as plain text. Use
    this when a question references a YouTube video and you need to know
    what is said in it — this returns only the spoken/captioned words, not
    any visual information from the video.

    Pass either the full YouTube URL as given in the question, or a bare
    video ID if that's what you have — both are handled automatically.

    If the result indicates captions are disabled or no transcript was
    found, that video genuinely has no available transcript — do not
    retry the same call. Instead, try web_search for information about
    the video's content (e.g. its title, description, or discussion of
    it elsewhere) as a fallback.
    """

    # pull the video ID out whether they gave a full URL or just the ID
    match = re.search(r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})", url_or_id)
    video_id = match.group(1) if match else url_or_id

    try:
        fetched = ytt_api.fetch(video_id)
        full_text = " ".join(snippet.text for snippet in fetched.snippets)
        return full_text
    except TranscriptsDisabled:
        return "This video has captions disabled — no transcript available."
    except NoTranscriptFound:
        return "No transcript found for this video."
    except Exception as e:
        return f"Error fetching transcript: {e}"
