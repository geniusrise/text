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

    CLI Usage:

    ```bash
        genius HuggingFaceSentimentAnalysisFineTuner rise \
            batch \
                --input_s3_bucket geniusrise-test \
                --input_s3_folder train \
            batch \
                --output_s3_bucket geniusrise-test \
                --output_s3_folder model \
            fine_tune \
                --args model_name=my_model tokenizer_name=my_tokenizer num_train_epochs=3 per_device_train_batch_size=8
    ```

    YAML Configuration:

    ```yaml
        version: "1"
        bolts:
            my_fine_tuner:
                name: "HuggingFaceSentimentAnalysisFineTuner"
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

            if self.data_extractor_lambda:
                fn = eval(self.data_extractor_lambda)
                data = [fn(d) for d in data]
            else:
                data = data

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
