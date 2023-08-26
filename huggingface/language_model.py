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

import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq
import yaml
from datasets import Dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling

from .base import HuggingFaceBatchFineTuner


class HuggingFaceLanguageModelingFineTuner(HuggingFaceBatchFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on language modeling tasks.

    ```
    Args:
        model: The pre-trained model to fine-tune.
        tokenizer: The tokenizer associated with the model.
        input_config (BatchInput): The batch input configuration.
        output_config (OutputConfig): The output configuration.
        state_manager (State): The state manager.
    ```
    """

    def load_dataset(self, dataset_path, **kwargs):
        r"""
        Load a language modeling dataset from a directory.

        ```
        The directory can contain any of the following file types:
        - Dataset files saved by the Hugging Face datasets library.
        - JSONL files: Each line is a JSON object representing an example. Structure:
            {
                "text": "The text content"
            }
        - CSV files: Should contain a 'text' column.
        - Parquet files: Should contain a 'text' column.
        - JSON files: Should be an array of objects with a 'text' key.
        - XML files: Each 'record' element should contain a 'text' child element.
        - YAML/YML files: Each document should be a dictionary with a 'text' key.
        - TSV files: Should contain a 'text' column separated by tabs.
        - Excel files (.xls, .xlsx): Should contain a 'text' column.
        - SQLite files (.db): Should contain a table with a 'text' column.
        - Feather files: Should contain a 'text' column.
        ```

        Args:
            dataset_path (str): The path to the dataset directory.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.
        """
        try:
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
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
                            text = record.find("text").text
                            data.append({"text": text})

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
                        query = "SELECT text FROM dataset_table;"
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
            self.log.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def prepare_train_features(self, examples):
        """
        Tokenize the examples and prepare the features for training.

        Args:
            examples (dict): A dictionary of examples.

        Returns:
            dict: The processed features.
        """
        # Extract the text from each examplep
        print(examples)
        texts = [example["text"] for example in examples["text"]]

        # Tokenize the examples
        tokenized_inputs = self.tokenizer(texts, truncation=True, padding=False)

        # Include the labels in the returned dictionary
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]

        return tokenized_inputs

    def data_collator(self, examples):
        """
        Customize the data collator.

        Args:
            examples: The examples to collate.

        Returns:
            dict: The collated data.
        """
        return DataCollatorForLanguageModeling(self.tokenizer, model=self.model)(examples)
