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

from typing import Any, List, Optional, Dict
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
import pandas as pd
import uuid
import yaml  # type: ignore
from pyarrow import feather
from pyarrow import parquet as pq
import glob
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
        self.log.info(f"Loading dataset from {dataset_path}")
        try:
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
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
                            text = record.find("text").text.split()  # type: ignore
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

                return Dataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def recognize_entities(
        self,
        model_name: str,
        max_length: int = 512,
        model_class: str = "AutoModelForSeq2SeqLM",
        tokenizer_class: str = "AutoTokenizer",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = True,
        awq_enabled: bool = False,
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
        self.awq_enabled = awq_enabled
        self.batch_size = batch_size

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
            awq_enabled=self.awq_enabled,
            **self.model_args,
        )

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        dataset = self.load_dataset(dataset_path)
        if dataset is None:
            self.log.error("Failed to load dataset.")
            return
        if dataset:
            dataset = dataset["text"]

        # Process data in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            if next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            predictions = self.model(**inputs, **generation_args)
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions.logits
            predictions = predictions.argmax(dim=-1).squeeze().tolist()

            self._save_predictions(inputs["input_ids"].tolist(), predictions, batch, output_path, i)

    def _save_predictions(
        self, inputs: list, predictions: list, input_batch: List[str], output_path: str, batch_idx: int
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
        label_predictions = [
            [
                {
                    "label": self.model.config.id2label[label_id],
                    "position": i,
                    "token": self.tokenizer.convert_ids_to_tokens(inp[i]),
                }
                for i, label_id in enumerate(pred)
            ]
            for pred, inp in zip(predictions, inputs)
        ]

        # Prepare data for saving
        data_to_save = [
            {"input": input_text, "labels": label} for input_text, label in zip(input_batch, label_predictions)
        ]
        with open(os.path.join(output_path, f"predictions-{batch_idx}-{str(uuid.uuid4())}.jsonl"), "w") as f:
            for item in data_to_save:
                f.write(json.dumps(item) + "\n")

        self.log.info(f"Saved predictions for batch {batch_idx} to {output_path}")
