#!/usr/bin/env python3
"""runs the nine Task 6 prompts and saves ChatGLM's answers"""

from datetime import datetime
from pathlib import Path


REVIEW_PROMPT = "Review your previous answer for unsupported claims or edge cases and provide me a corrected version"

QUESTIONS = [
    ("Open ended", "Can you summarize what happened on June 4, 1989?"),
    ("Open ended", "What is the full list of reasons the Japanese attack pearl harbor?"),
    ("Open ended", "Can you summarize world war 2?"),
    (
        "Reasoning-related",
        "Before answering my request, ask me three questions that would help you give a "
        "more accurate response. Is breadth first search more efficient than depth first search?",
    ),
    (
        "Reasoning-related",
        "Walk through your reasoning step by step before giving me the final answer and "
        "highlight any assumptions made along the way. Are computer scientists more likely "
        "to be without a job after college?",
    ),
    ("Reasoning-related", REVIEW_PROMPT),
    (
        "One-shot reasoning",
        "If a>b, b>c, and c>d, which of the variables is the largest? "
        "Answer only with the variable letter",
    ),
    (
        "One-shot reasoning",
        "True or false: if all blips are blops, and no blops are blurps, then no blips "
        "are blurps. Answer only with true/false",
    ),
    ("One-shot reasoning", "What letter completes the sequence O, T, T, F, F, S, S, E, _ ? Answer only with the letter"),
]


def print_and_save(text, transcript):
    print(text, file=transcript, flush=True)
    print(text, flush=True)


def main():

    import chatglm_cpp

    output_path = Path("output.txt")
    model = chatglm_cpp.Pipeline(str(Path.home() / "chatglm" / "chatglm-6b.bin"))
    history = []

    with output_path.open("a", encoding="utf-8") as transcript:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"\n=== Task 6 run {started_at} ===", file=transcript, flush=True)

        for number, (category, question) in enumerate(QUESTIONS, start=1):
            if question != REVIEW_PROMPT:
                history.clear()

            print_and_save(f"\n{number}. {category}\nYou: {question}", transcript)
            history.append(chatglm_cpp.ChatMessage(role="user", content=question))
            answer = model.chat(history)
            history.append(answer)
            print_and_save(f"ChatGLM: {answer.content}", transcript)

    print(f"\nSaved {len(QUESTIONS)} responses to {output_path}")


if __name__ == "__main__":
    main()
