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

import json
import os
import sqlite3
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml  # type: ignore
from datasets import Dataset, load_from_disk, load_dataset, load_metric
from geniusrise import BatchInput, BatchOutput, State
from pyarrow import feather
from pyarrow import parquet as pq

from geniusrise_text.base import TextBulk


class TextClassificationEval(TextBulk):
    r"""
    TextClassificationEval extends TextBulk to support evaluation of text classification models on large datasets. It facilitates
    processing of datasets, model inference, and computation of evaluation metrics such as accuracy, precision, recall, and F1 score.

    Args:
        input (BatchInput): Configuration and data inputs for the batch process.
        output (BatchOutput): Configurations for output data handling.
        state (State): State management for the classification task.
        **kwargs: Arbitrary keyword arguments for extended configurations.

    Example CLI Usage:
    ```bash
    genius TextClassificationEval rise \
        batch \
            --input_folder ./input \
        batch \
            --output_folder ./output \
        none \
        --id cardiffnlp/twitter-roberta-base-hate-multiclass-latest-lol \
        classify \
            --args \
                model_name="cardiffnlp/twitter-roberta-base-hate-multiclass-latest" \
                model_class="AutoModelForSequenceClassification" \
                tokenizer_class="AutoTokenizer" \
                use_cuda=True \
                precision="bfloat16" \
                quantization=0 \
                device_map="auto" \
                max_memory=None \
                torchscript=False
    ```
    """

    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        """
        Initializes the TextClassificationEval class with configurations for input, output, state, and evaluation settings.

        Args:
            input (BatchInput): Configuration for the input data.
            output (BatchOutput): Configuration for the output data.
            state (State): State management for the classification task.
            **kwargs: Additional keyword arguments for extended functionality.
        """
        super().__init__(input, output, state, **kwargs)

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs) -> Optional[Dataset]:
        r"""
        Load a classification dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            max_length (int, optional): The maximum length for tokenization. Defaults to 512.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"text": "The text content", "label": "The label"}
        ```

        ### CSV
        Should contain 'text' and 'label' columns.
        ```csv
        text,label
        "The text content","The label"
        ```

        ### Parquet
        Should contain 'text' and 'label' columns.

        ### JSON
        An array of dictionaries with 'text' and 'label' keys.
        ```json
        [{"text": "The text content", "label": "The label"}]
        ```

        ### XML
        Each 'record' element should contain 'text' and 'label' child elements.
        ```xml
        <record>
            <text>The text content</text>
            <label>The label</label>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'text' and 'label' keys.
        ```yaml
        - text: "The text content"
        label: "The label"
        ```

        ### TSV
        Should contain 'text' and 'label' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'text' and 'label' columns.

        ### SQLite (.db)
        Should contain a table with 'text' and 'label' columns.

        ### Feather
        Should contain 'text' and 'label' columns.
        """
        self.max_length = max_length

        self.label_to_id = self.model.config.label2id if self.model and self.model.config.label2id else {}  # type: ignore

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            if self.use_huggingface_dataset:
                dataset = load_dataset(self.huggingface_dataset)
            elif os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):

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
                            text = record.find("text").text  # type: ignore
                            label = record.find("label").text  # type: ignore
                            data.append({"text": text, "label": label})

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
                        query = "SELECT text, label FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                dataset = Dataset.from_pandas(pd.DataFrame(data))

            if hasattr(self, "map_data") and self.map_data:
                fn = eval(self.map_data)  # type: ignore
                dataset = dataset.map(fn)
            else:
                dataset = dataset

            return dataset
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def evaluate(
        self,
        model_name: str,
        model_class: str = "AutoModelForSequenceClassification",
        tokenizer_class: str = "AutoTokenizer",
        use_cuda: bool = False,
        precision: str = "float",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = False,
        compile: bool = False,
        awq_enabled: bool = False,
        flash_attention: bool = False,
        batch_size: int = 32,
        notification_email: Optional[str] = None,
        use_huggingface_dataset: bool = False,
        huggingface_dataset: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Evaluates the model on the loaded dataset, calculates evaluation metrics, and saves both predictions and metrics.

        Args:
            model_name (str): Name or path of the model.
            model_class (str): Class name of the model (default "AutoModelForSequenceClassification").
            tokenizer_class (str): Class name of the tokenizer (default "AutoTokenizer").
            use_cuda (bool): Whether to use CUDA for model inference (default False).
            precision (str): Precision for model computation (default "float").
            quantization (int): Level of quantization for optimizing model size and speed (default 0).
            device_map (str | Dict | None): Specific device to use for computation (default "auto").
            max_memory (Dict): Maximum memory configuration for devices.
            torchscript (bool, optional): Whether to use a TorchScript-optimized version of the pre-trained language model. Defaults to False.
            compile (bool, optional): Whether to compile the model before fine-tuning. Defaults to True.
            awq_enabled (bool): Whether to enable AWQ optimization (default False).
            flash_attention (bool): Whether to use flash attention optimization (default False).
            batch_size (int): Number of classifications to process simultaneously (default 32).
            use_huggingface_dataset (bool, optional): Whether to load a dataset from huggingface hub.
            huggingface_dataset (str, optional): The huggingface dataset to use.
            **kwargs: Arbitrary keyword arguments for model and generation configurations.
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
        self.compile = compile
        self.awq_enabled = awq_enabled
        self.flash_attention = flash_attention
        self.batch_size = batch_size
        self.notification_email = notification_email
        self.use_huggingface_dataset = use_huggingface_dataset
        self.huggingface_dataset = huggingface_dataset

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
            flash_attention=self.flash_attention,
            compile=self.compile,
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

        # Ensure metrics are available
        accuracy_metric = load_metric("accuracy")
        precision_metric = load_metric("precision")
        recall_metric = load_metric("recall")
        f1_metric = load_metric("f1")

        all_predictions = []
        all_true_labels = []

        # Loop through the dataset in batches
        for i in range(0, len(dataset), batch_size):
            batch_texts = dataset[i : i + batch_size]["text"]
            batch_labels = dataset[i : i + batch_size]["labels"]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            )

            if self.use_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
                batch_labels = torch.tensor(batch_labels).cuda()

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(batch_labels.cpu().numpy())

        # Compute overall metrics
        # fmt: off
        overall_accuracy = accuracy_metric.compute(predictions=all_predictions, references=all_true_labels)["accuracy"]
        overall_precision = precision_metric.compute(predictions=all_predictions, references=all_true_labels, average="macro")["precision"]
        overall_recall = recall_metric.compute(predictions=all_predictions, references=all_true_labels, average="macro")["recall"]
        overall_f1 = f1_metric.compute(predictions=all_predictions, references=all_true_labels, average="macro")["f1"]
        # fmt: on

        overall_evaluation_metrics = {
            "accuracy": overall_accuracy,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        }

        # Save predictions and evaluation metrics
        self._save_predictions(
            all_predictions, dataset["text"], all_true_labels, self.output.output_folder, overall_evaluation_metrics
        )

        self.done()

    def _save_predictions(
        self,
        predictions: List[int],
        input_texts: List[str],
        true_labels: List[int],
        output_path: str,
        evaluation_metrics: Dict[str, float],
    ) -> None:
        """
        Saves the classification predictions and evaluation metrics to a specified output path.

        Args:
            predictions (List[int]): List of label indices predicted by the model.
            input_texts (List[str]): List of original texts that were classified.
            true_labels (List[int]): List of true label indices.
            output_path (str): Path to save the classification results and metrics.
            evaluation_metrics (Dict[str, float]): Dictionary of evaluation metrics.
        """
        # Prepare data for saving
        data_to_save = [
            {"input": input_text, "prediction": prediction, "true_label": true_label}
            for input_text, prediction, true_label in zip(input_texts, predictions, true_labels)
        ]

        # Save predictions to a JSON file
        predictions_file_path = os.path.join(output_path, f"predictions-{str(uuid.uuid4())}.json")
        with open(predictions_file_path, "w") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

        # Save evaluation metrics to a JSON file
        metrics_file_path = os.path.join(output_path, "evaluation_metrics.json")
        with open(metrics_file_path, "w") as f:
            json.dump(evaluation_metrics, f, ensure_ascii=False, indent=4)
