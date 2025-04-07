# comprehension-bench

This is a tool for replicating the MK1 Comprehension Benchmark results presented in [the accompanying blog](PLACEHOLDER_URL).

The included dataset consists of a single generated logically consistent natural language document and with 100 associated queries on the document. Because these documents can be easily regenerated, model's cannot be trained on the dataset.

The document is made up of nested clauses formatted as follows:

```
The canyon is equipped for hands-on training if at least one of the following is satisfied:
    (1) All of the following must be true:
        (a) the tambourine is licensed for special operations
        (b) the talisman is proficient in energy conversion
``` 

Each query, which is provably true or false, is formatted as follows:

```
Given these conditions:
- the canyon is equipped for hands-on training
- the loom is synchronized with atomic clocks
- the hydra is aligned with strategic goals
Is the windmill consistently fashionably late?
```

The document's context length is extended by inserting clauses that are irrelevant to the original set. The benchmark consists of passing each query along with the document to a generation model, and then evaluating its response versus the known answer (true or false).

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
