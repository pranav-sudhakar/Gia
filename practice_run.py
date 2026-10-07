import os
import re
import requests
from build_model import react_graph
from langchain_core.messages import HumanMessage
import time
from groq import APIStatusError

def extract_final_answer(text: str) -> str:
    match = re.search(r"FINAL ANSWER:\s*(.+)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

HF_DATASET_BASE = "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation"
HF_TOKEN = os.environ['HF_TOKEN']

def download_file(api_url: str, task_id: str, file_name: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    urls_to_try = []
    if file_name:
        urls_to_try.append(f"{HF_DATASET_BASE}/{file_name}")
    urls_to_try.append(f"{api_url}/files/{task_id}")

    for url in urls_to_try:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            local_path = f"downloaded_{file_name}"
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded from {url} ({len(response.content)} bytes)")
            return local_path
        except requests.exceptions.HTTPError as e:
            print(f"Failed: {url} — status {e.response.status_code}: {e.response.text[:200]}")
            continue
    print(f"Could not download file for task {task_id} — proceeding without it")
    return None

def get_file_type(file_name: str) -> str:
    return file_name.split(".")[-1].lower()

def invoke_with_retry(graph, state, config, max_retries=3):
    for attempt in range(max_retries):
        try:
            return graph.invoke(state, config=config)
        except APIStatusError as e:
            if e.status_code == 413 and attempt < max_retries - 1:
                print(f"Rate limited, waiting 60s before retry {attempt + 1}/{max_retries}...")
                time.sleep(60)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

def run_gaia_question(api_url: str, question_data: dict) -> str:
    try:
        question = question_data["question"]
        task_id = question_data["task_id"]
        file_name = question_data.get("file_name", "")

        file_path, file_type = None, None
        if file_name:
            file_path = download_file(api_url, task_id, file_name)
            file_type = get_file_type(file_name)

        initial_state = {
            "messages": [HumanMessage(content=question)],
            "file_path": file_path,
            "file_type": file_type,
        }

        final_state = invoke_with_retry(react_graph, initial_state, {"recursion_limit": 20})
        last_msg = final_state["messages"][-1]
        print("RAW CONTENT:", repr(last_msg.content))
        print("STOP REASON:", getattr(last_msg, "response_metadata", {}).get("stop_reason"))
        raw_response = final_state["messages"][-1].text
        answer = extract_final_answer(raw_response)

        print(f"\nTASK ID: {task_id}")
        print(f"QUESTION: {question}")
        if file_name:
            print(f"ATTACHED FILE: {file_name}")
        print(f"AGENT ANSWER: {answer}")
        print("-" * 60)

        return answer
    except Exception as e:
        task_id = question_data.get("task_id", "unknown")
        print(f"\n[SKIPPED] Task {task_id} failed with: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()   # <- prints the full stack, including which line/function raised it
        print("-" * 60)
        return None

AGENT_CODE = "https://huggingface.co/spaces/your-username/gaia-agent-submission/tree/main"
USERNAME = "pranavsudhakar9"
API_URL = "https://agents-course-unit4-scoring.hf.space"

def run_full_submission():
    all_questions = requests.get(f"{API_URL}/questions").json()
    answers_payload = []

    for i, q in enumerate(all_questions, 1):
        print(f"\n=== {i}/{len(all_questions)} — {q['task_id']} ===")
        answer = run_gaia_question(API_URL, q)
        if answer is not None:
            answers_payload.append({
                "task_id": q["task_id"],
                "submitted_answer": answer,
            })
        else:
            print("Skipped — no answer submitted for this task")

    print(f"\nSubmitting {len(answers_payload)}/{len(all_questions)} answers...")
    submit_response = requests.post(f"{API_URL}/submit", json={
        "username": USERNAME,
        "agent_code": AGENT_CODE,
        "answers": answers_payload,
    })
    print(submit_response.json())

if __name__ == "__main__":
    run_full_submission()

"""
if __name__ == "__main__":
    all_questions = requests.get(f"{API_URL}/questions").json()
    print(f"Total questions available: {len(all_questions)}")
    test_batch = all_questions[:5]   # first 10 — change this slice as needed

    for i, question_data in enumerate(test_batch, 1):
        print(f"\n=== Question {i}/{len(test_batch)} ===")
        run_gaia_question(API_URL, question_data)

    target_id = "8e867cd7-cff9-4e6c-867a-ff5ddc2550be"
    question_data = next(q for q in all_questions if q["task_id"] == target_id)

    run_gaia_question(API_URL, question_data)
    """