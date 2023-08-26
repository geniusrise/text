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

from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase
import glob
import json
import logging
from typing import Dict

# import pandas as pd
import torch

from .base import HuggingFaceBatchFineTuner


class InstructionTuningDataset:
    def __init__(self, dir_path: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.data = []
        for file_path in glob.glob(f"{dir_path}/*"):
            if file_path.endswith(".jsonl"):
                with open(file_path, "r") as file:
                    self.data.extend([json.loads(line) for line in file])
            elif file_path.endswith(".arrow"):
                dataset = HFDataset.from_file(file_path)
                self.data.extend(dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Loaded {len(self.data)} examples from {dir_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        encoding = self.tokenizer(
            example["instruction"],
            example["output"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding


class HuggingFaceInstructionTuningFineTuner(HuggingFaceBatchFineTuner):
    def load_dataset(self, dataset_path: str, **kwargs):
        try:
            logging.info(f"Loading dataset from {dataset_path}")
            return InstructionTuningDataset(dataset_path, self.tokenizer)
        except Exception as e:
            logging.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise
