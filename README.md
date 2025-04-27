# Procedural Sentence Classification Using Multi-Agent Systems and Fine-Tuning

This repository contains multiple approaches for classifying sentences as **procedural** or **non-procedural**, focusing on surgical or instructional steps.  
We implement both **multi-agent reasoning** systems and **fine-tuning of a large language model** (QLoRA on Mistral-7B) to achieve high classification accuracy.

---

## Project Structure

### 1. Multi-Agent Procedural Classification
Using [CrewAI](https://github.com/joaomdmoura/crewAI) and [LangChain](https://github.com/langchain-ai/langchain), we build a multi-agent system with the following roles:
- **Summarizer Agent**: Condenses long procedural paragraphs into single sentences.
- **Classifier Agent**: Classifies the summarized sentence using few-shot learning.
- **Contextual Validator Agent**: Validates the classification using a Retrieval-Augmented Generation (RAG) system.
- **Final Validator Agent**: Combines classifier and contextual outputs into a final structured JSON.

Multiple variants are implemented:
- **Full Multi-Agent System** (Summarizer + Classifier + Context Validator + Final Validator)
- **Simplified Agents** (Only Classifier + Validator)
- **Increased Few-Shot Examples** for boosting performance.

### 2. Fine-Tuning with QLoRA
We fine-tune [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) using QLoRA to classify procedural sentences:
- Converts each sentence into a formatted prompt.
- Trains using [TRL's SFTTrainer](https://huggingface.co/docs/trl) for supervised fine-tuning.
- Two fine-tuning versions are provided:
  - **Simple Fine-Tuning**
  - **Fine-Tuning with Early Stopping and Optimized Hyperparameters**

---

## Dataset

The dataset (`SPKS.txt`) is a text file with labeled sentences:
- `__label__1` → Procedural (`yes`)
- `__label__2` → Non-Procedural (`no`)

Sentences are split and preprocessed into training and test sets.

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Setup

Set your API keys (for OpenAI and Hugging Face) in the environment variables or directly inside the script.

Choose your preferred approach:
- **Multi-Agent Approach**: Uses CrewAI workflows with agents and tasks.
- **Fine-Tuning Approach**: Fine-tunes and evaluates a language model on your data.

---

## Requirements

- Python 3.10+

**Libraries:**
- `langchain`
- `crewai`
- `openai`
- `faiss-cpu`
- `transformers`
- `datasets`
- `trl`
- `peft`
- `accelerate`
- `scikit-learn`
- `matplotlib`
- `ipywidgets`
- `bitsandbytes`

(See `requirements.txt` for the full dependency list.)

---

## Results

The system reports classification metrics including:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Both the agent-based approach and the fine-tuned model demonstrate strong performance in identifying procedural sentences.

---

## Future Work

- Further specialization of agent roles and task decomposition.
- Fine-tuning larger or domain-specific models for enhanced performance.
- Integrating explainability modules for better understanding of decisions.
- Expanding the dataset with more diverse procedural instructions.

---

## Credits

Developed by [Your Name Here].  
Inspired by CrewAI and recent advances in lightweight fine-tuning techniques (QLoRA) for large language models.

