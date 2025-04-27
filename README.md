Procedural Sentence Classification Using Multi-Agent Systems and Fine-Tuning
This repository contains multiple approaches for classifying sentences as procedural or non-procedural, focusing on surgical or instructional steps.
We implement both multi-agent reasoning systems and fine-tuning of a large language model (QLoRA on Mistral-7B) to achieve high classification accuracy.

Project Structure
1. Multi-Agent Procedural Classification
Using CrewAI and LangChain, we build a multi-agent system with the following roles:

Summarizer Agent: Condenses long procedural paragraphs into single sentences.

Classifier Agent: Classifies the summarized sentence using few-shot learning.

Contextual Validator Agent: Validates the classification using a Retrieval-Augmented Generation (RAG) system.

Final Validator Agent: Combines classifier and contextual outputs into a final structured JSON.

Multiple variants are implemented:

Full Multi-Agent System (Summarizer + Classifier + Context Validator + Final Validator)

Simplified Agents (Only Classifier + Validator)

Increased Few-Shot Examples for boosting performance.

2. Fine-Tuning with QLoRA
We fine-tune Mistral-7B using QLoRA to classify procedural sentences:

Converts each sentence into a formatted prompt.

Trains using TRL's SFTTrainer for supervised fine-tuning.

Two fine-tuning versions are provided:

Simple Fine-Tuning

Fine-Tuning with Early Stopping and Optimized Hyperparameters

Dataset
The dataset (SPKS.txt) is a text file with labeled sentences:

__label__1 → Procedural (yes)

__label__2 → Non-Procedural (no)

Sentences are split and preprocessed into training and test sets.

How to Run
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set your API keys (for OpenAI and Hugging Face) in the environment variables or in the script.

Run the notebook or Python script depending on your target method:

Multi-Agent Approach: Uses CrewAI workflows.

Fine-Tuning Approach: Fine-tunes and evaluates a model on your data.

Requirements
Python 3.10+

Libraries:

langchain

crewai

openai

faiss-cpu

transformers

datasets

trl

peft

accelerate

scikit-learn

matplotlib

ipywidgets

bitsandbytes

(See requirements.txt for full list.)

Results
The system reports classification metrics including accuracy, precision, recall, and F1-score.
Both the agent-based and fine-tuned models achieve strong results in detecting procedural content.

Future Work
Further agent specialization and task division.

Fine-tuning larger models or using different datasets.

Adding error analysis and explainability for predictions.

Credits
Developed by Nasim Bayat Chaleshtari.
Inspired by CrewAI and recent advances in lightweight fine-tuning (QLoRA) for LLMs.
