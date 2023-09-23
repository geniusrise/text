# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq
import yaml  # type: ignore
from datasets import Dataset, load_from_disk, load_metric
from nltk.translate.bleu_score import corpus_bleu
from transformers import DataCollatorForLanguageModeling, EvalPrediction

from geniusrise_huggingface.base import HuggingFaceFineTuner


class HuggingFaceLanguageModelingFineTuner(HuggingFaceFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on language modeling tasks.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    CLI Usage:

    ```bash
        genius HuggingFaceLanguageModelingFineTuner rise \
            batch \
                --input_s3_bucket geniusrise-test \
                --input_s3_folder train \
            batch \
                --output_s3_bucket geniusrise-test \
                --output_s3_folder model \
            fine_tune \
                --args model_name=my_model tokenizer_name=my_tokenizer num_train_epochs=3 per_device_train_batch_size=8 data_max_length=512
    ```

    YAML Configuration:

    ```yaml
        version: "1"
        bolts:
            my_fine_tuner:
                name: "HuggingFaceLanguageModelingFineTuner"
                method: "fine_tune"
                args:
                    model_name: "my_model"
                    tokenizer_name: "my_tokenizer"
                    num_train_epochs: 3
                    per_device_train_batch_size: 8
                    data_max_length: 512
                input:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        folder: "my_dataset"
                output:
                    type: "batch"
                    args:
                        bucket: "my_bucket"
                        folder: "my_model"
                deploy:
                    type: k8s
                    args:
                        kind: deployment
                        name: my_fine_tuner
                        context_name: arn:aws:eks:us-east-1:genius-dev:cluster/geniusrise-dev
                        namespace: geniusrise
                        image: geniusrise/geniusrise
                        kube_config_path: ~/.kube/config
    ```

    Supported Data Formats:
        - JSONL
        - CSV
        - Parquet
        - JSON
        - XML
        - YAML
        - TSV
        - Excel (.xls, .xlsx)
        - SQLite (.db)
        - Feather
    """

    def load_dataset(self, dataset_path, masked: bool = True, max_length: int = 512, **kwargs):
        r"""
        Load a language modeling dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            masked (bool, optional): Whether to use masked language modeling. Defaults to True.
            max_length (int, optional): The maximum length for tokenization. Defaults to 512.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### Dataset files saved by Hugging Face datasets library
        The directory should contain 'dataset_info.json' and other related files.

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"text": "The text content"}
        ```

        ### CSV
        Should contain 'text' column.
        ```csv
        text
        "The text content"
        ```

        ### Parquet
        Should contain 'text' column.

        ### JSON
        An array of dictionaries with 'text' key.
        ```json
        [{"text": "The text content"}]
        ```

        ### XML
        Each 'record' element should contain 'text' child element.
        ```xml
        <record>
            <text>The text content</text>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'text' key.
        ```yaml
        - text: "The text content"
        ```

        ### TSV
        Should contain 'text' column separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'text' column.

        ### SQLite (.db)
        Should contain a table with 'text' column.

        ### Feather
        Should contain 'text' column.
        """

        self.masked = masked
        self.max_length = max_length

        try:
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                dataset = load_from_disk(dataset_path)
            else:
                data = []
                for filename in os.listdir(dataset_path):
                    filepath = os.path.join(dataset_path, filename)
                    if filename.endswith(".jsonl"):
                        with open(filepath, "r") as f:
                            for line in f:
                                example = json.loads(line)
                                data.append(example)

                    elif filename.endswith(".csv"):
                        df = pd.read_csv(filepath)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".parquet"):
                        df = pq.read_table(filepath).to_pandas()
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".json"):
                        with open(filepath, "r") as f:
                            json_data = json.load(f)
                            data.extend(json_data)

                    elif filename.endswith(".xml"):
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for record in root.findall("record"):
                            text = record.find("text").text  # type: ignore
                            data.append({"text": text})

                    elif filename.endswith(".yaml") or filename.endswith(".yml"):
                        with open(filepath, "r") as f:
                            yaml_data = yaml.safe_load(f)
                            data.extend(yaml_data)

                    elif filename.endswith(".tsv"):
                        df = pd.read_csv(filepath, sep="\t")
                        data.extend(df.to_dict("records"))

                    elif filename.endswith((".xls", ".xlsx")):
                        df = pd.read_excel(filepath)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".db"):
                        conn = sqlite3.connect(filepath)
                        query = "SELECT text FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                if self.data_extractor_lambda:
                    fn = eval(self.data_extractor_lambda)
                    data = [fn(d) for d in data]
                else:
                    data = data

                dataset = Dataset.from_pandas(pd.DataFrame(data))

            # Preprocess the dataset
            if self.tokenizer and self.tokenizer.pad_token_id is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenized_dataset = dataset.map(
                self.prepare_train_features,
                batched=True,
                remove_columns=dataset.column_names,
            )

            return tokenized_dataset
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def prepare_train_features(self, examples):
        """
        Tokenize the examples and prepare the features for training.

        Args:
            examples (dict): A dictionary of examples.

        Returns:
            dict: The processed features.
        """
        # Tokenize the examples
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

        # Include the labels in the returned dictionary
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]

        return tokenized_inputs

    def data_collator(self, examples):
        """
        Customize the data collator.

        Args:
            examples: The examples to collate.

        Returns:
            dict: The collated data.
        """
        return DataCollatorForLanguageModeling(self.tokenizer, mlm=self.masked)(examples)

    def compute_metrics(self, eval_pred: EvalPrediction) -> Optional[Dict[str, float]]:
        """
        Compute evaluation metrics for the model's predictions.

        This method takes the model's predictions and ground truth labels, converts them to text,
        and then computes the BLEU score for evaluation.

        Args:
            eval_pred (EvalPrediction): A named tuple containing `predictions` and `label_ids`.
                - `predictions`: The logits predicted by the model of shape (batch_size, sequence_length, num_classes).
                - `label_ids`: The ground truth labels of shape (batch_size, sequence_length).

        Returns:
            Optional[Dict[str, float]]: A dictionary containing the BLEU score. Returns None if an exception occurs.

        Raises:
            Exception: If the tokenizer is not initialized.
        """

        predictions, labels = eval_pred
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        labels = labels[0] if isinstance(labels, tuple) else labels

        # Get the most likely token IDs from the logits (predictions)
        predictions = np.argmax(predictions, axis=1)

        # Convert labels and predictions to text
        if self.tokenizer:
            if len(labels.shape) == 1:
                labels = labels.reshape(-1, 1)
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)

            labels_text = [
                [self.tokenizer.decode(label, skip_special_tokens=True) for label in example] for example in labels
            ]
            predictions_text = [
                [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in example] for example in predictions
            ]

            # Flatten the lists to compute sacrebleu
            flat_labels_text = [" ".join(example) for example in labels_text]
            flat_predictions_text = [" ".join(example) for example in predictions_text]
        else:
            raise Exception("No tokenizer found, how did we even get here, please raise a PR.")

        # Compute BLEU score using sacrebleu
        sacrebleu_score = load_metric("sacrebleu").compute(
            predictions=flat_predictions_text,
            references=[[ref] for ref in flat_labels_text],
        )
        # Compute BLEU score
        bleu_score = corpus_bleu(labels_text, predictions_text)

        return {
            "sacrebleu": sacrebleu_score,
            "bleu": bleu_score,
        }
