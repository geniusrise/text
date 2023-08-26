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

from typing import Any, Dict, Union
import json
import os
import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import DataCollatorWithPadding

from .base import HuggingFaceBatchFineTuner


class HuggingFaceCommonsenseReasoningFineTuner(HuggingFaceBatchFineTuner):
    """
    A bolt for fine-tuning Hugging Face models on commonsense reasoning tasks.
    """

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Union[Dataset, DatasetDict]:
        """
        Load a dataset from a directory.

        Args:
            dataset_path (str): The path to the directory containing the dataset files.
            **kwargs: Additional keyword arguments to pass to the `load_dataset` method.

        Returns:
            Dataset: The loaded dataset.
        """
        try:
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                dataset = load_from_disk(dataset_path)
            else:
                # Load dataset from JSONL files
                data = []
                for filename in os.listdir(dataset_path):
                    if filename.endswith(".jsonl"):
                        with open(os.path.join(dataset_path, filename), "r") as f:
                            for line in f:
                                example = json.loads(line)
                                data.append(example)
                dataset = Dataset.from_pandas(pd.DataFrame(data))

            # Preprocess the dataset
            tokenized_dataset = dataset.map(
                self.prepare_train_features,
                batched=True,
                remove_columns=dataset.column_names,
            )

            return tokenized_dataset
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
