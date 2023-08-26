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

from datasets import load_from_disk, Dataset
from transformers import DataCollatorForLanguageModeling
import os
import json
import pandas as pd

from .base import HuggingFaceBatchFineTuner


class HuggingFaceLanguageModelingFineTuner(HuggingFaceBatchFineTuner):
    """
    A bolt for fine-tuning Hugging Face models on language modeling tasks.

    This bolt extends the HuggingFaceBatchFineTuner to handle the specifics of language modeling tasks,
    such as the specific format of the datasets and the specific metrics for evaluation.

    The dataset should be in the following format:
    - Each example is a dictionary with the following keys:
        - 'text': a string representing the text to be used for language modeling.
    """

    def load_dataset(self, dataset_path, **kwargs):
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
