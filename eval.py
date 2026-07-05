"""
Evaluation script for the RAG API.
Sends each test question to the running FastAPI server, checks whether
expected keywords appear in the answer, and reports an accuracy score.

Run this AFTER starting the server (uvicorn main:app --reload) in another terminal.
"""

import requests
import json

API_URL = "http://127.0.0.1:8000/ask"

# Each test case: the question, a list of keywords that SHOULD appear in
# a correct answer, and whether the correct answer is "I don't know" (refusal).
TEST_CASES = [
    {
        "question": "What is Harsh's current job title and company?",
        "expect_keywords": ["AI Engineer", "Quantum AI Research"],
        "should_refuse": False,
    },
    {
        "question": "What programming languages does Harsh know?",
        "expect_keywords": ["Dart", "Java", "Python"],
        "should_refuse": False,
    },
    {
        "question": "What certification does Harsh have?",
        "expect_keywords": ["Generative AI Mastermind"],
        "should_refuse": False,
    },
    {
        "question": "What is Harsh's CGPA and university?",
        "expect_keywords": ["7.2", "J.C. Bose"],
        "should_refuse": False,
    },
    {
        "question": "What SDK did Harsh use for the Video Chat Application?",
        "expect_keywords": ["Jitsi"],
        "should_refuse": False,
    },
    {
        "question": "What was Harsh's role at Garioxtech and what did he work on there?",
        "expect_keywords": ["Software Developer Intern", "Flutter"],
        "should_refuse": False,
    },
    {
        "question": "What payment gateway integrations has Harsh implemented, and in which projects?",
        "expect_keywords": ["Stripe", "Razorpay"],
        "should_refuse": False,
    },
    {
        "question": "What is Harsh's salary expectation?",
        "expect_keywords": [],
        "should_refuse": True,
    },
    {
        "question": "Has Harsh worked with Kubernetes or Docker?",
        "expect_keywords": [],
        "should_refuse": True,
    },
    {
        "question": "What programming languages does Harsh dislike?",
        "expect_keywords": [],
        "should_refuse": True,
    },
]

REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not mentioned",
    "cannot find",
    "no information",
]


def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


def check_keywords(answer: str, keywords: list) -> tuple:
    answer_lower = answer.lower()
    found = [kw for kw in keywords if kw.lower() in answer_lower]
    missing = [kw for kw in keywords if kw.lower() not in answer_lower]
    return found, missing


def run_eval(runs_per_question: int = 1):
    results = []
    total_checks = 0
    passed_checks = 0

    for case in TEST_CASES:
        question = case["question"]
        for run_num in range(1, runs_per_question + 1):
            try:
                resp = requests.post(API_URL, json={"question": question}, timeout=120)
                resp.raise_for_status()
                answer = resp.json()["answer"]
            except Exception as e:
                answer = f"[REQUEST FAILED: {e}]"

            refused = is_refusal(answer)

            if case["should_refuse"]:
                passed = refused
                found, missing = [], []
            else:
                found, missing = check_keywords(answer, case["expect_keywords"])
                passed = len(missing) == 0 and not refused

            total_checks += 1
            if passed:
                passed_checks += 1

            results.append({
                "question": question,
                "run": run_num,
                "answer": answer,
                "should_refuse": case["should_refuse"],
                "refused": refused,
                "expected_keywords": case["expect_keywords"],
                "found_keywords": found,
                "missing_keywords": missing,
                "passed": passed,
            })

            status = "PASS" if passed else "FAIL"
            print(f"[{status}] (run {run_num}) {question}")
            if not passed:
                print(f"       -> answer: {answer[:200]}")
                if missing:
                    print(f"       -> missing keywords: {missing}")

    accuracy = passed_checks / total_checks * 100 if total_checks else 0
    print(f"\n{'='*50}")
    print(f"Accuracy: {passed_checks}/{total_checks} ({accuracy:.1f}%)")
    print(f"{'='*50}")

    with open("eval_results.json", "w") as f:
        json.dump({
            "accuracy_percent": round(accuracy, 1),
            "passed": passed_checks,
            "total": total_checks,
            "details": results
        }, f, indent=2)
    print("Full results saved to eval_results.json")

    return accuracy


if __name__ == "__main__":
    # Run each question 3 times to also measure consistency, not just correctness
    run_eval(runs_per_question=3)
