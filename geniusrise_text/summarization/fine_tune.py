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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, load_metric
from pyarrow import feather
from pyarrow import parquet as pq
from transformers import DataCollatorForSeq2Seq, EvalPrediction

from geniusrise_text.base import TextFineTuner


class SummarizationFineTuner(TextFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on summarization tasks.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    CLI Usage:

    ```bash
        genius SummarizationFineTuner rise \
            batch \
                --input_folder ./input \
            batch \
                --output_folder ./output \
            none \
            fine_tune \
                --args \
                    model_name=my_model \
                    tokenizer_name=my_tokenizer \
                    num_train_epochs=3 \
                    per_device_train_batch_size=8
    ```
    """

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Optional[DatasetDict]:
        r"""
        Load a dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset | DatasetDict: The loaded dataset.

        ## Supported Data Formats and Structures:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"text": "The text content", "summary": "The summary"}
        ```

        ### CSV
        Should contain 'text' and 'summary' columns.
        ```csv
        text,summary
        "The text content","The summary"
        ```

        ### Parquet
        Should contain 'text' and 'summary' columns.

        ### JSON
        An array of dictionaries with 'text' and 'summary' keys.
        ```json
        [{"text": "The text content", "summary": "The summary"}]
        ```

        ### XML
        Each 'record' element should contain 'text' and 'summary' child elements.
        ```xml
        <record>
            <text>The text content</text>
            <summary>The summary</summary>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'text' and 'summary' keys.
        ```yaml
        - text: "The text content"
          summary: "The summary"
        ```

        ### TSV
        Should contain 'text' and 'summary' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'text' and 'summary' columns.

        ### SQLite (.db)
        Should contain a table with 'text' and 'summary' columns.

        ### Feather
        Should contain 'text' and 'summary' columns.
        """

        try:
            if self.use_huggingface_dataset:
                dataset = load_dataset(self.huggingface_dataset)
            elif os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
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
                            document = record.find("document").text  # type: ignore
                            summary = record.find("summary").text  # type: ignore
                            data.append({"document": document, "summary": summary})
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
                        query = "SELECT document, summary FROM dataset_table;"
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

            tokenized_dataset = dataset.map(
                self.prepare_train_features,
                batched=True,
                remove_columns=dataset.column_names,
            )
            return tokenized_dataset

        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def prepare_train_features(self, examples: Dict[str, Union[str, List[str]]]) -> Optional[Dict[str, List[int]]]:
        """
        Tokenize the examples and prepare the features for training.

        Args:
            examples (dict): A dictionary of examples.

        Returns:
            dict: The processed features.
        """
        if not self.tokenizer:
            raise Exception("No tokenizer found, please call load_models first.")

        # Tokenize the examples
        try:
            tokenized_inputs = self.tokenizer(examples["document"], truncation=True, padding=False)
            tokenized_targets = self.tokenizer(examples["summary"], truncation=True, padding=False)
        except Exception as e:
            self.log.exception(f"Error tokenizing examples: {e}")
            raise

        # Prepare the labels
        tokenized_inputs["labels"] = tokenized_targets["input_ids"]

        return tokenized_inputs

    def data_collator(
        self, examples: List[Dict[str, Union[str, List[int]]]]
    ) -> Dict[str, Union[List[int], List[List[int]]]]:
        """
        Customize the data collator.

        Args:
            examples: The examples to collate.

        Returns:
            dict: The collated data.
        """
        return DataCollatorForSeq2Seq(self.tokenizer, model=self.model)(examples)

    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute ROUGE metrics.

        Args:
            pred (EvalPrediction): The predicted results.

        Returns:
            dict: A dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        if not self.tokenizer:
            raise Exception("No tokenizer found, please call load_models first.")

        rouge = load_metric("rouge")

        preds = pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        # Initialize lists to store the decoded predictions and labels
        decoded_preds = []
        decoded_labels = []

        # Process each example in the batch individually
        for prediction, label in zip(preds, pred.label_ids):
            # Convert the logits into token IDs by taking the argmax along the last dimension
            pred_id = np.argmax(prediction, axis=-1)

            # Decode the token IDs into text using the tokenizer
            decoded_pred = self.tokenizer.decode(pred_id, skip_special_tokens=True)
            decoded_label = self.tokenizer.decode(label, skip_special_tokens=True)

            # Add the decoded text to the lists
            decoded_preds.append(decoded_pred)
            decoded_labels.append(decoded_label)

        # Compute the ROUGE scores
        rouge_output = rouge.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "rouge1": rouge_output["rouge1"].mid.fmeasure,
            "rouge2": rouge_output["rouge2"].mid.fmeasure,
            "rougeL": rouge_output["rougeL"].mid.fmeasure,
        }
