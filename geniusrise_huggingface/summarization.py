# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml  # type: ignore
from datasets import DatasetDict, load_from_disk, load_metric, Dataset
from pyarrow import feather
from pyarrow import parquet as pq
from transformers import DataCollatorForSeq2Seq, EvalPrediction

from geniusrise_huggingface.base import HuggingFaceFineTuner


class HuggingFaceSummarizationFineTuner(HuggingFaceFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on summarization tasks.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    ## Using geniusrise to invoke via command line
    ```bash
    genius HuggingFaceSummarizationFineTuner rise \
        streaming \
            --input_kafka_topic summarization_data \
            --input_kafka_cluster_connection_string localhost:9094 \
            --input_kafka_consumer_group_id geniusrise \
        streaming \
            --output_kafka_topic summarization_results \
            --output_kafka_cluster_connection_string localhost:9094 \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise \
            --postgres_table state \
        load_dataset \
            --args dataset_path=my_dataset_path
    ```

    ## Using geniusrise to invoke via YAML file
    ```yaml
    version: "1"
    bolts:
        my_summarization_bolt:
            name: "HuggingFaceSummarizationFineTuner"
            method: "load_dataset"
            args:
                dataset_path: "my_dataset_path"
            input:
                type: "streaming"
                args:
                    input_topic: "summarization_data"
                    kafka_servers: "localhost:9094"
                    group_id: "geniusrise"
            output:
                type: "streaming"
                args:
                    output_topic: "summarization_results"
                    kafka_servers: "localhost:9094"
            state:
                type: "postgres"
                args:
                    postgres_host: "127.0.0.1"
                    postgres_port: 5432
                    postgres_user: "postgres"
                    postgres_password: "postgres"
                    postgres_database: "geniusrise"
                    postgres_table: "state"
            deploy:
                type: "k8s"
                args:
                    name: "my_summarization_bolt"
                    namespace: "default"
                    image: "my_summarization_bolt_image"
                    replicas: 1
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
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
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
