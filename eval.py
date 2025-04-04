from datasets import load_dataset
from model import Model
import argparse
from collections import defaultdict
import os

parser = argparse.ArgumentParser()

parser.add_argument("--generation_backend", type=str, required=True)
parser.add_argument("--generation_model", type=str, required=True)
parser.add_argument("--generation_api_key", type=str, required=True)
parser.add_argument("--evaluation_backend", type=str, default="openai")
parser.add_argument("--evaluation_model", type=str, default="gpt-4o")
parser.add_argument("--evaluation_api_key", type=str, required=True)
parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate for each context length. By default, all samples are evaluated.")
args = parser.parse_args()

generation_model = Model(backend=args.generation_backend, model=args.generation_model, api_key=args.generation_api_key)
evaluation_model = Model(backend=args.evaluation_backend, model=args.evaluation_model, api_key=args.evaluation_api_key)

test_split = load_dataset("comprehension-dataset", split="test")

generation_prompt_template = open("prompt_templates/generation_prompt_template.txt", "r").read()
eval_prompt_template = open("prompt_templates/eval_prompt_template.txt", "r").read()

results = defaultdict(list)

generation_failures = []

for i, example in enumerate(test_split):
    if args.num_samples is None or len(results[example['context_length']]) < args.num_samples:
        generation_prompt = generation_prompt_template.format(query=example["query"], context=example["context"])
        generation_response = generation_model.generate(generation_prompt)

        if generation_response is None:
            generation_failures.append(example["context_length"])
            continue

        eval_prompt = eval_prompt_template.format(query=example["query"], answer=example["answer"], response=generation_response)
        eval_response = evaluation_model.generate(eval_prompt)

        correct = eval_response.strip().lower() == "yes"

        results[example['context_length']].append(correct)

    # Print updated results after each example
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Completed example {i+1}/{len(test_split)}")
    print("\n----------------------------------\n")
    print(f"Query:")
    print(f"{example['query']}")
    print("\n----------------------------------\n")
    print(f"Model Response:")
    print(f"{generation_response.strip()}")
    print("\n----------------------------------\n")
    print(f"Evaluation Response:")
    print(f"{eval_response.strip()}")
    print("\n----------------------------------\n")
    print("Results:")
    for context_length, correct_list in sorted(results.items()):
        if len(correct_list) > 0:
            accuracy = sum(correct_list) * 100 / len(correct_list)
            print(f"Context length {context_length}: {accuracy:.2f}% ({sum(correct_list)}/{len(correct_list)})")

    print(f"Generation failures: {generation_failures}")
