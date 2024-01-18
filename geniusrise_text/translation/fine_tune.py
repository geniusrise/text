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

import ast
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Optional

import pandas as pd
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from pyarrow import feather
from pyarrow import parquet as pq
from transformers import DataCollatorForSeq2Seq

from geniusrise_text.base import TextFineTuner


class TranslationFineTuner(TextFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on translation tasks.

    ```
    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.
        **kwargs: Arbitrary keyword arguments for extended functionality.
    ```

    CLI Usage:

    ```bash
        genius TranslationFineTuner rise \
            batch \
                --input_s3_bucket geniusrise-test \
                --input_s3_folder input/trans \
            batch \
                --output_s3_bucket geniusrise-test \
                --output_s3_folder output/trans \
            postgres \
                --postgres_host 127.0.0.1 \
                --postgres_port 5432 \
                --postgres_user postgres \
                --postgres_password postgres \
                --postgres_database geniusrise\
                --postgres_table state \
            --id facebook/mbart-large-50-many-to-many-mmt-lol \
            fine_tune \
                --args \
                    model_name=my_model \
                    tokenizer_name=my_tokenizer \
                    num_train_epochs=3 \
                    per_device_train_batch_size=8 \
                    data_max_length=512
    ```
    """

    def load_dataset(
        self,
        dataset_path: str,
        max_length: int = 512,
        origin: str = "en",
        target: str = "fr",
        **kwargs: Any,
    ) -> Optional[DatasetDict]:
        r"""
        Load a dataset from a directory.

        ## Supported Data Formats and Structures for Translation Tasks:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {
            "translation": {
                "en": "English text",
                "fr": "French text"
            }
        }
        ```

        ### CSV
        Should contain 'en' and 'fr' columns.
        ```csv
        en,fr
        "English text","French text"
        ```

        ### Parquet
        Should contain 'en' and 'fr' columns.

        ### JSON
        An array of dictionaries with 'en' and 'fr' keys.
        ```json
        [
            {
                "en": "English text",
                "fr": "French text"
            }
        ]
        ```

        ### XML
        Each 'record' element should contain 'en' and 'fr' child elements.
        ```xml
        <record>
            <en>English text</en>
            <fr>French text</fr>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'en' and 'fr' keys.
        ```yaml
        - en: "English text"
          fr: "French text"
        ```

        ### TSV
        Should contain 'en' and 'fr' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'en' and 'fr' columns.

        ### SQLite (.db)
        Should contain a table with 'en' and 'fr' columns.

        ### Feather
        Should contain 'en' and 'fr' columns.

        Args:
            dataset_path (str): The path to the directory containing the dataset files.
            max_length (int, optional): The maximum length for tokenization. Defaults to 512.
            origin (str, optional): The origin language. Defaults to 'en'.
            target (str, optional): The target language. Defaults to 'fr'.
            **kwargs: Additional keyword arguments.

        Returns:
            DatasetDict: The loaded dataset.
        """
        self.max_length = max_length
        self.origin = origin
        self.target = target
        self.tokenizer.src_lang = self.origin

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
                            origin = record.find(self.origin).text  # type: ignore
                            target = record.find(self.target).text  # type: ignore
                            data.append({"translation": {self.origin: origin, self.target: target}})
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
                        query = f"SELECT {self.origin}, {self.target} FROM dataset_table;"
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

    def prepare_train_features(self, examples):
        """
        Tokenize the examples and prepare the features for training.

        Args:
            examples (dict): A dictionary of examples.

        Returns:
            dict: The processed features.
        """

        # Tokenize the examples
        if "translation" in examples:
            examples["translation"] = [ast.literal_eval(e) if type(e) is str else e for e in examples["translation"]]

            origins = [x[self.origin] for x in examples["translation"]]
            targets = [x[self.target] for x in examples["translation"]]
        elif self.origin in examples:
            origins = examples[self.origin]
            targets = examples[self.target]

        tokenized_inputs = self.tokenizer(
            origins,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        tokenized_targets = self.tokenizer(
            targets,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        # Replace padding token id by -100
        labels = [
            [(lbl if lbl != self.tokenizer.pad_token_id else -100) for lbl in label]
            for label in tokenized_targets["input_ids"]
        ]

        # Prepare the labels
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def data_collator(self, examples):
        """
        Customize the data collator.

        Args:
            examples: The examples to collate.

        Returns:
            dict: The collated data.
        """
        return DataCollatorForSeq2Seq(self.tokenizer, model=self.model)(examples)
