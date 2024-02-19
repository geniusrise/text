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

import glob
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml  # type: ignore
from datasets import Dataset, load_dataset, load_from_disk
from nltk.translate.bleu_score import corpus_bleu
from pyarrow import feather
from transformers import EvalPrediction

from geniusrise_text.base import TextFineTuner


class InstructionFineTuner(TextFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on instruction tuning tasks.

    This class inherits from `TextFineTuner` and specializes in fine-tuning models for instruction-based tasks.
    It provides additional methods for loading and preparing datasets in various formats, as well as computing custom metrics.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    Attributes:
        max_length (int): The maximum length for tokenization.

    CLI Usage:

    ```bash
        genius InstructionFineTuner rise \
            batch \
                --input_folder ./input \
            batch \
                --output_folder ./output \
            none \
            --id mistralai/Mistral-7B-Instruct-v0.1-lol \
            fine_tune \
                --args \
                    model_name=my_model \
                    tokenizer_name=my_tokenizer \
                    num_train_epochs=3 \
                    per_device_train_batch_size=8 \
                    data_max_length=512
    ```
    """

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs: Any) -> Union[Dataset, Dict]:
        r"""
        Load an instruction tuning dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            max_length (int, optional): The maximum length for tokenization. Defaults to 512.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"instruction": "The instruction", "output": "The output"}
        ```

        ### CSV
        Should contain 'instruction' and 'output' columns.
        ```csv
        instruction,output
        "The instruction","The output"
        ```

        ### Parquet
        Should contain 'instruction' and 'output' columns.

        ### JSON
        An array of dictionaries with 'instruction' and 'output' keys.
        ```json
        [{"instruction": "The instruction", "output": "The output"}]
        ```

        ### XML
        Each 'record' element should contain 'instruction' and 'output' child elements.
        ```xml
        <record>
            <instruction>The instruction</instruction>
            <output>The output</output>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'instruction' and 'output' keys.
        ```yaml
        - instruction: "The instruction"
          output: "The output"
        ```

        ### TSV
        Should contain 'instruction' and 'output' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'instruction' and 'output' columns.

        ### SQLite (.db)
        Should contain a table with 'instruction' and 'output' columns.

        ### Feather
        Should contain 'instruction' and 'output' columns.
        """

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            self.max_length = max_length
            if self.use_huggingface_dataset:
                dataset = load_dataset(self.huggingface_dataset)
            elif os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                dataset = load_from_disk(dataset_path)
            else:
                data = []
                for filename in glob.glob(f"{dataset_path}/*"):
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
                            instruction = record.find("instruction").text  # type: ignore
                            output = record.find("output").text  # type: ignore
                            data.append({"instruction": instruction, "output": output})

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
                        query = "SELECT instruction, output FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                dataset = Dataset.from_pandas(pd.DataFrame(data))

            if hasattr(self, "map_data") and self.map_data:
                fn = eval(self.map_data)  # type: ignore
                dataset = dataset.map(fn)
            else:
                dataset = dataset

            return dataset.map(self.prepare_train_features, batched=True)
        except Exception as e:
            self.log.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def prepare_train_features(self, examples: Dict) -> Dict:
        """
        Tokenize the examples and prepare the features for training.

        Args:
            examples (dict): A dictionary of examples.

        Returns:
            dict: The processed features.
        """
        try:
            if not self.tokenizer:
                raise Exception("Tokenizer not initialized")

            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Tokenize the instructions and outputs separately
            input_encoding = self.tokenizer(
                examples["instruction"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length // 2,
                return_tensors="pt",
            )
            output_encoding = self.tokenizer(
                examples["output"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length // 2,
                return_tensors="pt",
            )

            sep_token = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id else self.tokenizer.eos_token_id
            # Convert sep_token to a tensor, then expand dimensions
            sep_token_tensor = torch.tensor([sep_token], dtype=torch.long).unsqueeze(0)

            input_ids = torch.cat([input_encoding["input_ids"], sep_token_tensor, output_encoding["input_ids"]], dim=1)
            attention_mask = torch.cat(
                [
                    input_encoding["attention_mask"],
                    torch.tensor([[1]], dtype=torch.long),
                    output_encoding["attention_mask"],
                ],
                dim=1,
            )

            # Use the tokenized output as the labels
            labels = output_encoding["input_ids"]

            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        except Exception as e:
            self.log.exception(f"Error preparing train features: {e}")
            raise

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
        else:
            raise Exception("No tokenizer found, how did we even get here, please raise a PR.")

        # Compute BLEU score
        bleu_score = corpus_bleu(labels_text, predictions_text)

        return {
            "bleu": bleu_score,
        }
