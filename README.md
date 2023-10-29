<h1 align="center">
  <img src="./assets/logo_with_text.png" alt=logo" width="900"/>
</h1>
<h2 align="center">
  <a style="color:#f34960" href="https://docs.geniusrise.ai">Documentation</a>
</h2>

<p align="center">
  <img src="https://img.shields.io/github/actions/workflow/status/geniusrise/geniusrise-text/pytest.yml?branch=master" alt="GitHub Workflow Status"/>
  <img src="https://codecov.io/gh/geniusrise/geniusrise-text/branch/main/graph/badge.svg?token=0b359b3a-f29c-4966-9661-a79386b3450d" alt="Codecov"/>
  <img src="https://img.shields.io/github/license/geniusrise/geniusrise-text" alt="Codecov"/>
  <img src="https://img.shields.io/github/issues/geniusrise/geniusrise-text" alt="Codecov"/>
</p>

---

## <span style="color:#e667aa">About</span>

<span style="color:#e4e48c">Geniusrise</span> is a modular, loosely-coupled
AgentOps / MLOps framework designed for the era of Large Language Models,
offering flexibility, inclusivity, and standardization in designing networks of
AI agents.

It seamlessly integrates tasks, state management, data handling, and model
versioning, all while supporting diverse infrastructures and user expertise
levels. With its plug-and-play architecture,
<span style="color:#e4e48c">Geniusrise</span> empowers teams to build, share,
and deploy AI agent workflows across various platforms.

## <span style="color:#e667aa">Huggingface Bolts</span>

This is a collection of generic streaming and (micro) batch bolts interfacing
with the huggingface ecosystem.

Includes:

| No. | Name                                                  | Description                                    | Input Type | Output Type |
| --- | ----------------------------------------------------- | ---------------------------------------------- | ---------- | ----------- |
| 1   | [Text Classification](#text-classification)           | Fine-tuning for text classification tasks      | Batch      | Batch       |
| 2   | [Instruction Tuning](#instruction-tuning)             | Fine-tuning for instruction tuning tasks       | Batch      | Batch       |
| 3   | [Commonsense Reasoning](#commonsense-reasoning)       | Fine-tuning for commonsense reasoning tasks    | Batch      | Batch       |
| 4   | [Language Modeling](#language-modeling)               | Fine-tuning for language modeling tasks        | Batch      | Batch       |
| 5   | [Named Entity Recognition](#named-entity-recognition) | Fine-tuning for named entity recognition tasks | Batch      | Batch       |
| 6   | [Question Answering](#question-answering)             | Fine-tuning for question answering tasks       | Batch      | Batch       |
| 7   | [Sentiment Analysis](#sentiment-analysis)             | Fine-tuning for sentiment analysis tasks       | Batch      | Batch       |
| 8   | [Summarization](#summarization)                       | Fine-tuning for summarization tasks            | Batch      | Batch       |
| 9   | [Translation](#translation)                           | Fine-tuning for translation tasks              | Batch      | Batch       |
