import requests

API_URL = "CONFIRMED_URL_FROM_TEMPLATE"
AGENT_CODE = "https://huggingface.co/spaces/your-username/gaia-agent-submission/tree/main"
USERNAME = "your-hf-username"

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