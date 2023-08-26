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

from typing import Any, Optional
import os
import json
import pandas as pd
from pyarrow import parquet as pq
from pyarrow import feather
import sqlite3
import yaml
import xml.etree.ElementTree as ET
from datasets import DatasetDict, load_from_disk
from transformers import DataCollatorForSeq2Seq

from .base import HuggingFaceBatchFineTuner


class HuggingFaceTranslationFineTuner(HuggingFaceBatchFineTuner):
    """
    A bolt for fine-tuning Hugging Face models on translation tasks.

    This bolt extends the HuggingFaceBatchFineTuner to handle the specifics of translation tasks,
    such as the specific format of the datasets and the specific metrics for evaluation.

    The dataset should be in the following format:
    - Each example is a dictionary with the following keys:
        - 'translation': a dictionary with two keys:
            - 'en': a string representing the English text.
            - 'fr': a string representing the French text.
    """

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Optional[DatasetDict]:
        """
        Load a dataset from a directory.

        The directory can contain any of the following file types:
        - Dataset files saved by the Hugging Face datasets library.
        - JSONL files: Each line is a JSON object representing an example. Structure:
            {
                "translation": {
                    "en": "English text",
                    "fr": "French text"
                }
            }
        - CSV files: Should contain 'en' and 'fr' columns.
        - Parquet files: Should contain 'en' and 'fr' columns.
        - JSON files: Should contain an array of objects with 'en' and 'fr' keys.
        - XML files: Each 'record' element should contain 'en' and 'fr' child elements.
        - YAML files: Each document should be a dictionary with 'en' and 'fr' keys.
        - TSV files: Should contain 'en' and 'fr' columns separated by tabs.
        - Excel files (.xls, .xlsx): Should contain 'en' and 'fr' columns.
        - SQLite files (.db): Should contain a table with 'en' and 'fr' columns.
        - Feather files: Should contain 'en' and 'fr' columns.

        Args:
            dataset_path (str): The path to the directory containing the dataset files.
            **kwargs: Additional keyword arguments.

        Returns:
            DatasetDict: The loaded dataset.
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
                            en = record.find("en").text  # type: ignore
                            fr = record.find("fr").text  # type: ignore
                            data.append({"translation": {"en": en, "fr": fr}})
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
                        query = "SELECT en, fr FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))
                dataset = DatasetDict({"train": pd.DataFrame(data)})

            tokenized_dataset = dataset.map(
                self.prepare_train_features,
                batched=True,
                remove_columns=dataset.column_names,
            )
            return tokenized_dataset

        except Exception as e:
            self.log.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            return None

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
            [x["en"] for x in examples["translation"]],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tokenized_targets = self.tokenizer(
            [x["fr"] for x in examples["translation"]],
            truncation=True,
            padding="max_length",
            max_length=512,
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
