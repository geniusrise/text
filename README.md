![banner](./assets/banner.jpg)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

# Huggingface Bolts

This is a collection of generic streaming and (micro) batch bolts interfacing
with the huggingface ecosystem.

**Table of Contents**

- [Huggingface Bolts](#huggingface-bolts)
  - [Usage](#usage)
  - [Usage](#usage-1)
    - [Text Classification](#text-classification)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

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

## Usage

To test, first bring up all related services via the supplied docker-compose:

```bash
docker compose up -d
docker compose logs -f
```

These management consoles will be available:

| Console  | Link                   |
| -------- | ---------------------- |
| Kafka UI | http://localhost:8088/ |

Postgres can be accessed with:

```bash
docker exec -it geniusrise-postgres-1 psql -U postgres
```

## Usage

### Text Classification

To fine-tune a model for text classification tasks, you can use the following
command:

```bash
genius HuggingFaceClassificationFineTuner rise \
  batch \
      --input_folder my_dataset \
  streaming \
      --output_kafka_topic my_topic \
      --output_kafka_cluster_connection_string localhost:9094 \
  postgres \
      --postgres_host 127.0.0.1 \
      --postgres_port 5432 \
      --postgres_user postgres \
      --postgres_password postgres \
      --postgres_database geniusrise \
      --postgres_table state \
  load_dataset \
      --args
```
