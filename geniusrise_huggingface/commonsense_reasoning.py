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
from typing import Any, Dict, Union

import pandas as pd
import pyarrow.parquet as pq
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_from_disk
from pyarrow import feather
from transformers import DataCollatorWithPadding

from geniusrise_huggingface.base import HuggingFaceFineTuner


class HuggingFaceCommonsenseReasoningFineTuner(HuggingFaceFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on commonsense reasoning tasks.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    ## Using geniusrise to invoke via command line
    ```bash
    genius HuggingFaceCommonsenseReasoningFineTuner rise \
        streaming \
            --input_kafka_topic commonsense_test \
            --input_kafka_cluster_connection_string localhost:9094 \
            --input_kafka_consumer_group_id commonsense_group \
        streaming \
            --output_kafka_topic commonsense_output \
            --output_kafka_cluster_connection_string localhost:9094 \
        load_dataset \
            --args dataset_path=my_dataset max_length=512
    ```

    ## Using geniusrise to invoke via YAML file
    ```yaml
    version: "1"
    bolts:
        my_commonsense_bolt:
            name: "HuggingFaceCommonsenseReasoningFineTuner"
            method: "load_dataset"
            args:
                dataset_path: "my_dataset"
                max_length: 512
            input:
                type: "streaming"
                args:
                    input_topic: "commonsense_test"
                    kafka_servers: "localhost:9094"
                    group_id: "commonsense_group"
            output:
                type: "streaming"
                args:
                    output_topic: "commonsense_output"
                    kafka_servers: "localhost:9094"
    ```
    """

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Union[Dataset, DatasetDict, None]:
        r"""
        Load a commonsense reasoning dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### Hugging Face Dataset
        Dataset files saved by the Hugging Face datasets library.

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"premise": "The premise text", "hypothesis": "The hypothesis text", "label": 0 or 1 or 2}
        ```

        ### CSV
        Should contain 'premise', 'hypothesis', and 'label' columns.
        ```csv
        premise,hypothesis,label
        "The premise text","The hypothesis text",0
        ```

        ### Parquet
        Should contain 'premise', 'hypothesis', and 'label' columns.

        ### JSON
        An array of dictionaries with 'premise', 'hypothesis', and 'label' keys.
        ```json
        [{"premise": "The premise text", "hypothesis": "The hypothesis text", "label": 0}]
        ```

        ### XML
        Each 'record' element should contain 'premise', 'hypothesis', and 'label' child elements.
        ```xml
        <record>
            <premise>The premise text</premise>
            <hypothesis>The hypothesis text</hypothesis>
            <label>0</label>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'premise', 'hypothesis', and 'label' keys.
        ```yaml
        - premise: "The premise text"
          hypothesis: "The hypothesis text"
          label: 0
        ```

        ### TSV
        Should contain 'premise', 'hypothesis', and 'label' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'premise', 'hypothesis', and 'label' columns.

        ### SQLite (.db)
        Should contain a table with 'premise', 'hypothesis', and 'label' columns.

        ### Feather
        Should contain 'premise', 'hypothesis', and 'label' columns.
        """

        try:
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                dataset = load_from_disk(dataset_path)
                return dataset.map(
                    self.prepare_train_features,
                    batched=True,
                    remove_columns=dataset.column_names,
                )
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
                            data.extend(json.load(f))
                    elif filename.endswith(".xml"):
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for record in root.findall("record"):
                            example = {
                                "premise": record.find("premise").text,  # type: ignore
                                "hypothesis": record.find("hypothesis").text,  # type: ignore
                                "label": int(record.find("label").text),  # type: ignore
                            }
                            data.append(example)
                    elif filename.endswith((".yaml", ".yml")):
                        with open(filepath, "r") as f:
                            data.extend(yaml.safe_load(f))
                    elif filename.endswith(".tsv"):
                        df = pd.read_csv(filepath, sep="\t")
                        data.extend(df.to_dict("records"))
                    elif filename.endswith((".xls", ".xlsx")):
                        df = pd.read_excel(filepath)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".db"):
                        conn = sqlite3.connect(filepath)
                        query = "SELECT premise, hypothesis, label FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                dataset = Dataset.from_pandas(pd.DataFrame(data))
                return dataset.map(
                    self.prepare_train_features,
                    batched=True,
                    remove_columns=dataset.column_names,
                )

        except Exception as e:
            print(f"Error loading dataset: {e}")
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

            # Tokenize the examples
            tokenized_inputs = self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding=False,
            )

            # Prepare the labels
            tokenized_inputs["labels"] = examples["label"]

            return tokenized_inputs
        except Exception as e:
            print(f"Error preparing train features: {e}")
            raise

    def data_collator(self, examples: Dict) -> Dict:
        """
        Customize the data collator.

        Args:
            examples: The examples to collate.

        Returns:
            dict: The collated data.
        """
        try:
            return DataCollatorWithPadding(self.tokenizer)(examples)

        except Exception as e:
            print(f"Error in data collation: {e}")
            raise
