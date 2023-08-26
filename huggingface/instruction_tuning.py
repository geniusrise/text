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

import glob
import json
import logging
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union

import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from datasets import Dataset as HFDataset
from pyarrow import feather
from transformers import PreTrainedTokenizerBase

from .base import HuggingFaceBatchFineTuner


class HuggingFaceInstructionTuningFineTuner(HuggingFaceBatchFineTuner):
    """
    A bolt for fine-tuning Hugging Face models on instruction tuning tasks.

    Args:
        model: The pre-trained model to fine-tune.
        tokenizer: The tokenizer associated with the model.
        input_config (BatchInput): The batch input configuration.
        output_config (OutputConfig): The output configuration.
        state_manager (State): The state manager.
    """

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Union[HFDataset, Dict]:
        """
        Load an instruction tuning dataset from a directory.

        The directory can contain any of the following file types:
        - Dataset files saved by the Hugging Face datasets library (.arrow).
        - JSONL files: Each line is a JSON object representing an example. Structure:
            {
                "instruction": "The instruction",
                "output": "The output"
            }
        - CSV files: Should contain 'instruction' and 'output' columns.
        - Parquet files: Should contain 'instruction' and 'output' columns.
        - JSON files: Should be an array of objects with 'instruction' and 'output' keys.
        - XML files: Each 'record' element should contain 'instruction' and 'output' child elements.
        - YAML/YML files: Each document should be a dictionary with 'instruction' and 'output' keys.
        - TSV files: Should contain 'instruction' and 'output' columns separated by tabs.
        - Excel files (.xls, .xlsx): Should contain 'instruction' and 'output' columns.
        - SQLite files (.db): Should contain a table with 'instruction' and 'output' columns.
        - Feather files: Should contain 'instruction' and 'output' columns.

        Args:
            dataset_path (str): The path to the dataset directory.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.
        """
        try:
            logging.info(f"Loading dataset from {dataset_path}")
            data = []
            for filename in glob.glob(f"{dataset_path}/*"):
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
                        instruction = record.find("instruction").text  # type: ignore
                        output = record.find("output").text  # type: ignore
                        data.append({"instruction": instruction, "output": output})

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
                    query = "SELECT instruction, output FROM dataset_table;"
                    df = pd.read_sql_query(query, conn)
                    data.extend(df.to_dict("records"))

                elif filename.endswith(".feather"):
                    df = feather.read_feather(filepath)
                    data.extend(df.to_dict("records"))

            return HFDataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            logging.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise
