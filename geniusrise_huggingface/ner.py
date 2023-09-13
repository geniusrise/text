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

import ast
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq
import torch
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DataCollatorForTokenClassification, EvalPrediction

from geniusrise_huggingface.base import HuggingFaceFineTuner


class HuggingFaceNamedEntityRecognitionFineTuner(HuggingFaceFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on named entity recognition tasks.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    ## Using geniusrise to invoke via command line
    ```bash
    genius HuggingFaceNamedEntityRecognitionFineTuner rise \
        batch \
            --input_bucket my_bucket \
            --input_folder my_folder \
        batch \
            --output_bucket my_output_bucket \
            --output_folder my_output_folder \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise \
            --postgres_table state \
        load_dataset \
            --args dataset_path=my_dataset_path label_list="['O', 'B-PER', 'I-PER']"
    ```

    ## Using geniusrise to invoke via YAML file
    ```yaml
    version: "1"
    bolts:
        my_ner_bolt:
            name: "HuggingFaceNamedEntityRecognitionFineTuner"
            method: "load_dataset"
            args:
                dataset_path: "my_dataset_path"
                label_list: ["O", "B-PER", "I-PER"]
            input:
                type: "batch"
                args:
                    bucket: "my_bucket"
                    folder: "my_folder"
            output:
                type: "batch"
                args:
                    bucket: "my_output_bucket"
                    folder: "my_output_folder"
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
                    name: "my_ner_bolt"
                    namespace: "default"
                    image: "my_ner_bolt_image"
                    replicas: 1
    ```
    """

    def load_dataset(
        self, dataset_path: str, label_list: List[str] = [], **kwargs: Any
    ) -> Union[Dataset, DatasetDict, None]:
        r"""
        Load a named entity recognition dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            label_list (List[str], optional): The list of labels for named entity recognition. Defaults to [].

        Returns:
            DatasetDict: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### Hugging Face Dataset
        Dataset files saved by the Hugging Face datasets library.

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"tokens": ["token1", "token2", ...], "ner_tags": [0, 1, ...]}
        ```

        ### CSV
        Should contain 'tokens' and 'ner_tags' columns.
        ```csv
        tokens,ner_tags
        "['token1', 'token2', ...]", "[0, 1, ...]"
        ```

        ### Parquet
        Should contain 'tokens' and 'ner_tags' columns.

        ### JSON
        An array of dictionaries with 'tokens' and 'ner_tags' keys.
        ```json
        [{"tokens": ["token1", "token2", ...], "ner_tags": [0, 1, ...]}]
        ```

        ### XML
        Each 'record' element should contain 'tokens' and 'ner_tags' child elements.
        ```xml
        <record>
            <tokens>token1 token2 ...</tokens>
            <ner_tags>0 1 ...</ner_tags>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'tokens' and 'ner_tags' keys.
        ```yaml
        - tokens: ["token1", "token2", ...]
          ner_tags: [0, 1, ...]
        ```

        ### TSV
        Should contain 'tokens' and 'ner_tags' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'tokens' and 'ner_tags' columns.

        ### SQLite (.db)
        Should contain a table with 'tokens' and 'ner_tags' columns.

        ### Feather
        Should contain 'tokens' and 'ner_tags' columns.
        """

        self.label_list = label_list
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
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
                    # Additional file types support
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
                            tokens = record.find("tokens").text.split()  # type: ignore
                            ner_tags = list(map(int, record.find("ner_tags").text.split()))  # type: ignore
                            data.append({"tokens": tokens, "ner_tags": ner_tags})
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
                        query = "SELECT tokens, ner_tags FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                dataset = Dataset.from_pandas(pd.DataFrame(data))

            # Preprocess the dataset
            tokenized_dataset = dataset.map(
                self.prepare_train_features,
                batched=True,
                remove_columns=dataset.column_names,
            )

            return tokenized_dataset
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def prepare_train_features(
        self, examples: Dict[str, Union[List[str], List[int]]]
    ) -> Dict[str, Union[List[str], List[int]]]:
        """
        Tokenize the examples and prepare the features for training.

        Args:
            examples (Dict[str, Union[List[str], List[int]]]): A dictionary of examples.

        Returns:
            Dict[str, Union[List[str], List[int]]]: The processed features.
        """

        if not self.tokenizer:
            raise Exception("Tokenizer and model have to be loaded first, please call load_models() first.")

        # convert into proper structure if coming from a csv etc
        examples["tokens"] = [
            ast.literal_eval(example) if type(example) is str else example for example in examples["tokens"]
        ]
        examples["ner_tags"] = [
            ast.literal_eval(example) if type(example) is str else example for example in examples["ner_tags"]
        ]

        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = []

        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            label_ids = []
            for word_idx in word_ids:
                if word_idx is not None:
                    print(f"label[word_idx]: {labels[word_idx]}", self.label_to_id)  # Debug print
                    label_ids.append(self.label_to_id[labels[word_idx]])  # type: ignore
                else:
                    label_ids.append(-100)
            all_labels.append(label_ids)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    def data_collator(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Customize the data collator.

        Args:
            examples (List[Dict[str, torch.Tensor]]): The examples to collate.

        Returns:
            Dict[str, torch.Tensor]: The collated data.
        """
        return DataCollatorForTokenClassification(self.tokenizer)(examples)

    def compute_metrics(self, eval_prediction: EvalPrediction, average: str = "micro"):
        predictions = np.argmax(eval_prediction.predictions, axis=-1)
        labels = eval_prediction.label_ids

        # Mask out ignored values
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average=average),
            "recall": recall_score(labels, predictions, average=average),
            "f1": f1_score(labels, predictions, average=average),
        }
