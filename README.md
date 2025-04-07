# comprehension-bench

This tool replicates the MK1 Comprehension Benchmark, designed to evaluate the reasoning capabilities of large language models. The benchmark and its results are detailed in [the accompanying blog post](PLACEHOLDER_URL).

The benchmark uses a dataset comprising a single, procedurally generated, logically consistent natural language document paired with 100 associated queries. Because the documents are uniquely generated each time, models cannot simply memorize the dataset; they must reason over the provided text.

The document consists of nested clauses structured like this example:

```
The canyon is equipped for hands-on training if at least one of the following is satisfied:
    (1) All of the following must be true:
        (a) the tambourine is licensed for special operations
        (b) the talisman is proficient in energy conversion
```

Each query presents a set of conditions based on the document and asks a question with a provably true or false answer, formatted like this:

```
Given these conditions:
- the canyon is equipped for hands-on training
- the loom is synchronized with atomic clocks
- the hydra is aligned with strategic goals
Is the windmill consistently fashionably late?
```

To assess model performance across varying input sizes, the document's context length is systematically increased by inserting clauses that are irrelevant to the core logical structure. The evaluation process involves sending the extended document and a query to a target model, then comparing the model's response against the known correct answer (true or false).

# Installation

To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

# Benchmark Results

The plot below shows the performance of various models on the benchmark across different context lengths.

![Benchmark Results](./assets/plot.png)

## Example Usage

`python eval.py --generation_model "gemini-2.0-flash" --generation_backend "google" --generation_api_key $GOOGLE_API_KEY --evaluation_api_key $OPENAI_API_KEY`

The `--num_samples` flag is optional. Set this to a small number (e.g. 5) to evaluate a small subset of the dataset, or leave it unset to evaluate the entire dataset.

## License
The code repository is licensed under the [MIT License](LICENSE).
