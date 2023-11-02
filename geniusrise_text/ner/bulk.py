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

from typing import Any, List, Optional
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
import pandas as pd
import uuid
import yaml  # type: ignore
from pyarrow import feather
from pyarrow import parquet as pq
import torch
from datasets import Dataset, load_from_disk
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_text.base import TextBulk


class NamedEntityRecognitionBulk(TextBulk):
    """
    A class for bulk Named Entity Recognition (NER) using Hugging Face models.
    """

    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs: Any) -> None:
        super().__init__(input, output, state, **kwargs)

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Optional[Dataset]:
        """
        Load a NER dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.
        """
        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                return load_from_disk(dataset_path)
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
                            tokens = record.find("tokens").text.split()  # type: ignore
                            data.append({"tokens": tokens})
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
                        query = "SELECT tokens FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                return Dataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def infer(
        self,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        """
        Perform NER inference on the loaded dataset.

        Args:
            batch_size (int): The batch size for inference.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        _dataset = self.load_dataset(dataset_path)
        if _dataset is None:
            self.log.error("Failed to load dataset.")
            return
        dataset = _dataset["text"]

        # Process data in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            if next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            predictions = self.model(**inputs)
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions
            if next(self.model.parameters()).is_cuda:
                predictions = predictions.cpu()

            self._save_predictions(predictions, batch, output_path, i)

    def _save_predictions(
        self, predictions: torch.Tensor, input_batch: List[str], output_path: str, batch_idx: int
    ) -> None:
        """
        Save the NER predictions to disk.

        Args:
            predictions (torch.Tensor): The NER predictions.
            input_batch (List[str]): The input batch.
            output_path (str): The output directory path.
            batch_idx (int): The batch index.

        Returns:
            None
        """
        # Convert tensor of label ids to list of label strings
        id_to_label = dict(enumerate(self.model.config.id2label.values()))  # type: ignore
        label_predictions = [[id_to_label[label_id] for label_id in pred] for pred in predictions.tolist()]

        # Prepare data for saving
        data_to_save = [
            {"input": input_text, "prediction": label} for input_text, label in zip(input_batch, label_predictions)
        ]
        with open(os.path.join(output_path, f"predictions-{batch_idx}-{str(uuid.uuid4())}.jsonl"), "w") as f:
            for item in data_to_save:
                f.write(json.dumps(item) + "\n")

        self.log.info(f"Saved predictions for batch {batch_idx} to {output_path}")
