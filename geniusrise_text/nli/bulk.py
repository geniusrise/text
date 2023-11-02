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

from typing import Any, Optional
import json
import os
import sqlite3
import xml.etree.ElementTree as ET

import pandas as pd
import pyarrow.parquet as pq
import yaml  # type: ignore
from datasets import Dataset
from pyarrow import feather
import torch
import uuid
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_text.base import TextBulk


class NLIBulk(TextBulk):
    """
    A class for bulk Natural Language Inference (NLI) using Hugging Face models.
    """

    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        super().__init__(input, output, state, **kwargs)

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs) -> Optional[Dataset]:
        """
        Load an NLI dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            max_length (int, optional): The maximum length for tokenization. Defaults to 512.
            **kwargs: Additional keyword arguments to pass to the underlying dataset loading functions.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        Supported Data Formats and Structures:
        ---------------------------------------

        The following data formats and structures are supported:

        - JSONL: Each line is a JSON object representing an example.
            {"premise": "The premise text", "hypothesis": "The hypothesis text"}

        - CSV: Should contain 'premise' and 'hypothesis' columns.
            premise,hypothesis
            "The premise text","The hypothesis text"

        - Parquet: Should contain 'premise' and 'hypothesis' columns.

        - JSON: An array of dictionaries with 'premise' and 'hypothesis' keys.
            [{"premise": "The premise text", "hypothesis": "The hypothesis text"}]

        - XML: Each 'record' element should contain a 'premise' and 'hypothesis' child elements.
            <record>
                <premise>The premise text</premise>
                <hypothesis>The hypothesis text</hypothesis>
            </record>

        - YAML: Each document should be a dictionary with 'premise' and 'hypothesis' keys.
            - premise: "The premise text"
              hypothesis: "The hypothesis text"

        - TSV: Should contain 'premise' and 'hypothesis' columns separated by tabs.

        - Excel (.xls, .xlsx): Should contain 'premise' and 'hypothesis' columns.

        - SQLite (.db): Should contain a table with 'premise' and 'hypothesis' columns.

        - Feather: Should contain 'premise' and 'hypothesis' columns.
        """

        self.max_length = max_length

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
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
                        premise = record.find("premise").text  # type: ignore
                        hypothesis = record.find("hypothesis").text  # type: ignore
                        data.append({"premise": premise, "hypothesis": hypothesis})

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
                    query = "SELECT premise, hypothesis FROM dataset_table;"
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
        Perform NLI inference on the loaded dataset.

        Args:
            batch_size (int, optional): The batch size for inference. Defaults to 32.
            **kwargs: Additional keyword arguments.
        """

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        _dataset = self.load_dataset(dataset_path)
        if _dataset is None:
            self.log.error("Failed to load dataset.")
            return
        dataset = _dataset["text"]

        # Process dataset
        dataset = dataset.map(
            lambda example: self.tokenizer(
                example["premise"],
                example["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            ),
            batched=True,
        )

        # Perform inference
        self.log.info(f"Performing NLI inference on {len(dataset)} examples.")
        predictions = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            inputs = self.tokenizer(batch["input_ids"], padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.tolist())

        # Save results
        self.log.info(f"Saving results to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"nli_results_{uuid.uuid4().hex}.jsonl")
        with open(output_file, "w") as f:
            for example, pred in zip(dataset, predictions):
                result = {
                    "premise": example["premise"],
                    "hypothesis": example["hypothesis"],
                    "prediction": self.id2label[pred],
                }
                f.write(json.dumps(result) + "\n")

        self.log.info("Inference completed.")
