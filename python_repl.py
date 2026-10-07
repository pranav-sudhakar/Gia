def python_repl(code: str) -> str:
    """Execute Python code and return whatever it prints. Use this for any
    math, counting, sorting, filtering, or data manipulation — never
    perform calculations yourself in reasoning text, always compute them
    here to avoid arithmetic errors.

    Only the Python standard library is available (math, statistics,
    itertools, collections, re, etc.) plus pandas, which is available for
    reading and analyzing spreadsheet files. Your code must end with a
    print() statement — the value of the last expression alone is not
    captured, only printed output is returned.

    If the code raises an error, you will get back a short error message
    describing what went wrong (not a full traceback). Read the error,
    fix the code, and try again — do not give up and guess the answer
    after a single failed attempt.
    """
    import io
    import contextlib

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": __builtins__})
        result = output.getvalue()
        return result if result else "Code ran with no printed output. Add a print() statement."
    except Exception as e:
        return f"Error: {e}"
