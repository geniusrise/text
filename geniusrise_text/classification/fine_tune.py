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
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import yaml  # type: ignore
from datasets import Dataset, load_dataset, load_from_disk
from pyarrow import feather
from pyarrow import parquet as pq
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, EvalPrediction

from geniusrise_text.base import TextFineTuner


class TextClassificationFineTuner(TextFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models for text classification tasks.

    This class extends the `TextFineTuner` and specializes in fine-tuning models for text classification.
    It provides additional functionalities for loading and preprocessing text classification datasets in various formats.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    CLI Usage:

    ```bash
    genius TextClassificationFineTuner rise \
        batch \
            --input_folder ./input \
        batch \
            --output_folder ./output \
        none \
        --id cardiffnlp/twitter-roberta-base-hate-multiclass-latest-lol \
            fine_tune \
                --args \
                    model_name=my_model \
                    tokenizer_name=my_tokenizer \
                    num_train_epochs=3 \
                    per_device_train_batch_size=8 \
                    data_max_length=512
    ```
    """

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs) -> Optional[Dataset]:
        r"""
        Load a classification dataset from a directory.

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
        {"text": "The text content", "label": "The label"}
        ```

        ### CSV
        Should contain 'text' and 'label' columns.
        ```csv
        text,label
        "The text content","The label"
        ```

        ### Parquet
        Should contain 'text' and 'label' columns.

        ### JSON
        An array of dictionaries with 'text' and 'label' keys.
        ```json
        [{"text": "The text content", "label": "The label"}]
        ```

        ### XML
        Each 'record' element should contain 'text' and 'label' child elements.
        ```xml
        <record>
            <text>The text content</text>
            <label>The label</label>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'text' and 'label' keys.
        ```yaml
        - text: "The text content"
        label: "The label"
        ```

        ### TSV
        Should contain 'text' and 'label' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'text' and 'label' columns.

        ### SQLite (.db)
        Should contain a table with 'text' and 'label' columns.

        ### Feather
        Should contain 'text' and 'label' columns.
        """

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, max_length=max_length)
        self.max_length = max_length

        # self.label_to_id = self.model.config.label2id if self.model and self.model.config.label2id else {}  # type: ignore
        self.label_to_id = {}

        def tokenize_function(examples):
            tokenized_data = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            labels = [x for x in list(set(examples["label"]))]
            all_labels = [l for l in examples["label"]]

            unknown_labels = [label for label in labels if label not in self.label_to_id]

            self.label_to_id = {
                **self.label_to_id,
                **{x: i for i, x in enumerate(unknown_labels)},
            }

            tokenized_data["label"] = [self.label_to_id[label] for label in all_labels]
            return tokenized_data

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            if self.use_huggingface_dataset:
                dataset = load_dataset(self.huggingface_dataset)
            elif os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
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
                            label = record.find("label").text  # type: ignore
                            data.append({"text": text, "label": label})

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
                        query = "SELECT text, label FROM dataset_table;"
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

            # Create label_to_id mapping and save it in model config
            # TODO: ugly shit cause we dont know num labels before we process the data but need tokenizer to process data
            self.label_to_id = {label: i for i, label in enumerate(set(dataset["train"]["label"]))}
            if self.model:
                config = self.model.config
                config.label2id = self.label_to_id
                config.id2label = {i: label for label, i in self.label_to_id.items()}
                config.num_labels = len(self.label_to_id.keys())
                self.config = config

                self.load_models(
                    model_name=self.model_name,
                    tokenizer_name=self.tokenizer_name,
                    model_class=self.model_class,
                    tokenizer_class=self.tokenizer_class,
                    device_map=self.device_map,
                    precision=self.precision,
                    quantization=self.quantization,
                    lora_config=self.lora_config,
                    use_accelerate=self.use_accelerate,
                    accelerate_no_split_module_classes=self.accelerate_no_split_module_classes,
                    **self.model_kwargs,
                )
            if self.tokenizer and not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

            self.log.info(self.model.config)

            return dataset.map(tokenize_function, batched=True)

        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def compute_metrics(self, eval_pred: EvalPrediction) -> Union[Optional[Dict[str, float]], Dict[str, float]]:
        """
        Compute metrics for evaluation. This class implements a simple classification evaluation,
        tasks should ideally override this.

        Args:
            eval_pred (EvalPrediction): The evaluation predictions.

        Returns:
            dict: The computed metrics.
        """
        predictions, labels = eval_pred
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        labels = labels[0] if isinstance(labels, tuple) else labels
        predictions = np.argmax(predictions, axis=1)

        is_binary = len(self.label_to_id.keys()) == 2
        average_type = "binary" if is_binary else "weighted"

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average_type)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
