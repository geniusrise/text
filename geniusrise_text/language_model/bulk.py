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

from typing import Any, Dict, Optional, List
import json
import os
import sqlite3
import glob
import xml.etree.ElementTree as ET

import pandas as pd
import yaml  # type: ignore
from datasets import Dataset, load_from_disk
from pyarrow import feather
from pyarrow import parquet as pq
import uuid
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_text.base import TextBulk


class LanguageModelBulk(TextBulk):
    """
    A class for bulk completion using Hugging Face models.

    Args:
        input (BatchInput): The input data to classify.
        output (BatchOutput): The output data to store the completion results.
        state (State): The state object to store intermediate state.
        **kwargs: Additional keyword arguments to pass to the base class.

    Attributes:
        max_length (int): The maximum length for tokenization.
        label_to_id (Dict[str, int]): A dictionary mapping label names to label IDs.

    Methods:
        load_dataset(dataset_path: str, max_length: int = 512, **kwargs) -> Optional[Dataset]:
            Load a completion dataset from a directory.

        classify(model_name: str, tokenizer_name: str, model_args: Optional[Dict[str, Any]] = None,
                 tokenizer_args: Optional[Dict[str, Any]] = None, batch_size: int = 32,
                 num_processes: int = 1, **kwargs) -> None:
            Classify the input data using the specified Hugging Face model and tokenizer.
    """

    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        super().__init__(input, output, state, **kwargs)

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs) -> Optional[Dataset]:
        r"""
        Load a completion dataset from a directory.

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
            {"text": "The text content"}

        - CSV: Should contain 'text' column.
            text
            "The text content"

        - Parquet: Should contain 'text' column.

        - JSON: An array of dictionaries with 'text' key.
            [{"text": "The text content"}]

        - XML: Each 'record' element should contain a 'text' child element.
            <record>
                <text>The text content</text>
            </record>

        - YAML: Each document should be a dictionary with 'text' key.
            - text: "The text content"

        - TSV: Should contain 'text' column separated by tabs.

        - Excel (.xls, .xlsx): Should contain 'text' column.

        - SQLite (.db): Should contain a table with 'text' column.

        - Feather: Should contain 'text' column.
        """

        self.max_length = max_length

        self.label_to_id = self.model.config.label2id if self.model and self.model.config.label2id else {}  # type: ignore

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                return load_from_disk(dataset_path)
            else:
                data = []
                for filename in glob.glob(f"{dataset_path}/**/*", recursive=True):
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
                            text = record.find("text").text  # type: ignore
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

                if hasattr(self, "map_data") and self.map_data:
                    fn = eval(self.map_data)  # type: ignore
                    data = [fn(d) for d in data]
                else:
                    data = data

                return Dataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def complete(
        self,
        model_name: str,
        model_class: str = "AutoModelForCausalLM",
        tokenizer_class: str = "AutoTokenizer",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = True,
        decoding_strategy: str = "generate",
        **kwargs: Any,
    ) -> None:
        if ":" in model_name:
            model_revision = model_name.split(":")[1]
            tokenizer_revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            tokenizer_name = model_name
        else:
            model_revision = None
            tokenizer_revision = None
            tokenizer_name = model_name

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.use_cuda = use_cuda
        self.precision = precision
        self.quantization = quantization
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript

        model_args = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}
        self.model_args = model_args

        generation_args = {k.replace("generation_", ""): v for k, v in kwargs.items() if "generation_" in k}
        self.generation_args = generation_args

        self.model, self.tokenizer = self.load_models(
            model_name=self.model_name,
            tokenizer_name=self.tokenizer_name,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            model_class=self.model_class,
            tokenizer_class=self.tokenizer_class,
            use_cuda=self.use_cuda,
            precision=self.precision,
            quantization=self.quantization,
            device_map=self.device_map,
            max_memory=self.max_memory,
            torchscript=self.torchscript,
            **self.model_args,
        )

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        _dataset = self.load_dataset(dataset_path)
        if _dataset is None:
            self.log.error("Failed to load dataset.")
            return
        dataset = _dataset["text"]

        prompts = []
        completions = []
        for _, prompt in enumerate(dataset):
            completion = self.generate(
                prompt=prompt,
                decoding_strategy=decoding_strategy,
                **generation_args,
            )
            completions.append(completion)
            prompts.append(prompt)

        self._save_completions(completions, prompts, output_path)

    def _save_completions(self, completions: List[str], prompts: List[str], output_path: str) -> None:
        # Prepare data for saving
        data_to_save = [
            {"prompt": prompt, "completion": completion} for prompt, completion in zip(prompts, completions)
        ]
        with open(os.path.join(output_path, f"completions-{str(uuid.uuid4())}.json"), "w") as f:
            json.dump(data_to_save, f)
