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
import logging
from typing import Optional
import pandas as pd
from datasets import Dataset, load_from_disk

from .base import HuggingFaceBatchFineTuner


class HuggingFaceClassificationFineTuner(HuggingFaceBatchFineTuner):
    """
    A bolt for fine-tuning Hugging Face models for text classification tasks.

    Args:
        model: The pre-trained model to fine-tune.
        tokenizer: The tokenizer associated with the model.
        input_config (BatchInput): The batch input configuration.
        output_config (OutputConfig): The output configuration.
        state_manager (State): The state manager.
    """

    def load_dataset(self, dataset_path: str, **kwargs) -> Optional[Dataset]:
        """
        Load a classification dataset from a directory.

        The directory can contain either:
        - Dataset files saved by the Hugging Face datasets library, or
        - JSONL files where each line is a JSON object representing an example. Each JSON object should have the
          following structure:
            {
                "text": "The text content",
                "label": "The label"
            }

        Args:
            dataset_path (str): The path to the dataset directory.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.
        """
        try:
            logging.info(f"Loading dataset from {dataset_path}")
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                return load_from_disk(dataset_path)
            else:
                # Load dataset from JSONL files
                data = []
                for filename in os.listdir(dataset_path):
                    if filename.endswith(".jsonl"):
                        with open(os.path.join(dataset_path, filename), "r") as f:
                            for line in f:
                                example = json.loads(line)
                                data.append(example)
                return Dataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            logging.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise
