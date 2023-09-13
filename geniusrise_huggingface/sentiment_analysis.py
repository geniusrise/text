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
from typing import Any, Dict, List, Union

import pandas as pd
import torch
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_from_disk
from pyarrow import feather
from pyarrow import parquet as pq
from transformers import DataCollatorWithPadding

from geniusrise_huggingface.base import HuggingFaceFineTuner


class HuggingFaceSentimentAnalysisFineTuner(HuggingFaceFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on sentiment analysis tasks.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    ## Using Command Line
    ```bash
    genius HuggingFaceSentimentAnalysisFineTuner rise \
        streaming \
            --input_kafka_topic webhook_test \
            --input_kafka_cluster_connection_string localhost:9094 \
            --input_kafka_consumer_group_id geniusrise \
        streaming \
            --output_kafka_topic webhook_test \
            --output_kafka_cluster_connection_string localhost:9094 \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise \
            --postgres_table state \
        listen \
            --args various=30 arguments=40 that=50 this=70 bolt=63 may=lol have='{"lol": "lel"}'
    ```

    ## Using YAML File
    ```yaml
    version: "1"
    bolts:
        my_fine_tuner:
            name: "HuggingFaceSentimentAnalysisFineTuner"
            method: "load_dataset"
            args:
                dataset_path: "/path/to/dataset"
            input:
                type: "batch"
                args:
                    bucket: "my-bucket"
                    folder: "my-folder"
            output:
                type: "streaming"
                args:
                    output_topic: "webhook_test"
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
                    name: "my_fine_tuner"
                    namespace: "default"
                    image: "my_fine_tuner_image"
                    replicas: 1
    ```

    Args:
        model: The pre-trained model to fine-tune.
        tokenizer: The tokenizer associated with the model.
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.
    """

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Dataset | DatasetDict:
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

        tokenized_dataset = dataset.map(
            self.prepare_train_features,
            batched=True,
            remove_columns=dataset.column_names,
        )
        return tokenized_dataset

    def prepare_train_features(self, examples: Dict[str, Union[str, int]]) -> Dict[str, Union[List[int], int]]:
        """
        Tokenize the examples and prepare the features for training.

        Args:
            examples (Dict[str, Union[str, int]]): A dictionary of examples.

        Returns:
            Dict[str, Union[List[int], int]]: The processed features.
        """
        if not self.tokenizer:
            raise Exception("No tokenizer found, please call load_models first.")

        tokenized_inputs = self.tokenizer(examples["text"], truncation=True, padding=False)
        tokenized_inputs["labels"] = examples["label"]
        return tokenized_inputs

    def data_collator(
        self, examples: List[Dict[str, Union[List[int], int]]]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Customize the data collator.

        Args:
            examples (List[Dict[str, Union[List[int], int]]]): The examples to collate.

        Returns:
            Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: The collated data.
        """
        return DataCollatorWithPadding(self.tokenizer)(examples)
