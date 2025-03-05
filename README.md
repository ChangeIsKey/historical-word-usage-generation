# Sense-Specific Historical Word Usage Generation

This repository provides resources and models for generating historical word usage examples tailored to specific senses of polysemous words. The aim is to assist in understanding how different meanings of a word have evolved over time by generating contextually appropriate sentences for each sense.

## Overview

Understanding the historical usage of words, especially those with multiple meanings, is crucial for linguistic research, lexicography, and natural language processing applications. This project addresses this need by offering models that generate sentences reflecting specific senses of polysemous words across different historical periods.

## Repository Structure
```
- README.md  # Documentation
- requirements.txt  # Dependencies
- datasets/
  - control_dataset.jsonl  # Control dataset for comparison
  - generated_usages/
    - Janus(PoS).jsonl
    - Janus.jsonl
    - Meta-Llama-3-70B-Instruct_1.0.jsonl
    - Meta-Llama-3-8B-Instruct_1.0.jsonl
    - few_shots_gpt4_predictions.jsonl
    - gpt4_predictions.jsonl
    - gpt_predictions.jsonl
- src/
  - example_of_janus_generation.py  # Example script for usage generation
  - decade_classification/classifier.py  # Decade classification model
  - gpt_predict/openai_api.py  # GPT-based predictions
  - llama_finetuning/finetuning.py  # Fine-tuning Llama models
  - llama_predict/
    - predict_finetuned.py  # Predictions using fine-tuned models
    - predict_instructed.py  # Predictions using instructed models
  - wsd_regression/classifier.py  # Word Sense Disambiguation classifier
```


## Available Models

The following models have been trained to perform sense-specific historical word usage generation and evaluate the output:

- **llama3-janus:** A text-to-text generation model designed for producing sense-specific historical sentences.
- **llama3-janus-pos:** An extension of llama3-janus which integrates part-of-speech tags.
- **text-dating:** A text classification model that determines the historical period of a given text.
- **graded-wsd:** A model focused on graded word sense disambiguation, providing nuanced sense distinctions in generated sentences.

These models are accessible through the Hugging Face Hub:

- [llama3-janus](https://huggingface.co/ChangeIsKey/llama3-janus)
- [llama3-janus-pos](https://huggingface.co/ChangeIsKey/llama3-janus-pos)
- [text-dating](https://huggingface.co/ChangeIsKey/text-dating)
- [graded-wsd](https://huggingface.co/ChangeIsKey/graded-wsd)

## Datasets
- **Control Dataset:** Contains baseline examples for comparison.
- **Generated Usages:** Includes outputs from various models such as Janus, Meta-Llama, and GPT.
- **Human Annotated usage-definition pairs:** [GWSD: A Graded Word Sense Disambiguation Dataset](https://zenodo.org/records/14974455)

## Citation

If you use these models or resources in your research, please cite the associated paper:

```bibtex
@article{cassotti2025,
  title     = {Sense-Specific Historical Word Usage Generation},
  authors    = {Pierluigi Cassotti, Nina Tahmasebi},
  journal = {TACL},
  year      = {2025}
}
```

## Acknowledgments

This work is part of the "Change is Key!" research program, which aims to create computational tools to explore the evolution of language, society, and culture. The program is funded by the Riksbankens Jubileumsfond under reference number M21-0021.

For more information, visit the [Change is Key! website](https://www.changeiskey.org/).
