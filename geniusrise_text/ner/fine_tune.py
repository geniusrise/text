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

import ast
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq
import torch
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DataCollatorForTokenClassification, EvalPrediction

from geniusrise_text.base import TextFineTuner


class NamedEntityRecognitionFineTuner(TextFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on named entity recognition tasks.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    CLI Usage:

    ```bash
        genius NamedEntityRecognitionFineTuner rise \
            batch \
                --input_folder ./input \
            batch \
                --output_folder ./output \
            none \
        --id dslim/bert-large-NER-lol \
            fine_tune \
                --args \
                    model_name=my_model \
                    tokenizer_name=my_tokenizer \
                    num_train_epochs=3 \
                    per_device_train_batch_size=8
    ```
    """

    def load_dataset(
        self, dataset_path: str, label_list: List[str] = [], **kwargs: Any
    ) -> Union[Dataset, DatasetDict, None]:
        r"""
        Load a named entity recognition dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            label_list (List[str], optional): The list of labels for named entity recognition. Defaults to [].

        Returns:
            DatasetDict: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### Hugging Face Dataset
        Dataset files saved by the Hugging Face datasets library.

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"tokens": ["token1", "token2", ...], "ner_tags": [0, 1, ...]}
        ```

        ### CSV
        Should contain 'tokens' and 'ner_tags' columns.
        ```csv
        tokens,ner_tags
        "['token1', 'token2', ...]", "[0, 1, ...]"
        ```

        ### Parquet
        Should contain 'tokens' and 'ner_tags' columns.

        ### JSON
        An array of dictionaries with 'tokens' and 'ner_tags' keys.
        ```json
        [{"tokens": ["token1", "token2", ...], "ner_tags": [0, 1, ...]}]
        ```

        ### XML
        Each 'record' element should contain 'tokens' and 'ner_tags' child elements.
        ```xml
        <record>
            <tokens>token1 token2 ...</tokens>
            <ner_tags>0 1 ...</ner_tags>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'tokens' and 'ner_tags' keys.
        ```yaml
        - tokens: ["token1", "token2", ...]
          ner_tags: [0, 1, ...]
        ```

        ### TSV
        Should contain 'tokens' and 'ner_tags' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'tokens' and 'ner_tags' columns.

        ### SQLite (.db)
        Should contain a table with 'tokens' and 'ner_tags' columns.

        ### Feather
        Should contain 'tokens' and 'ner_tags' columns.
        """

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
                    # Additional file types support
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
                            ner_tags = list(map(int, record.find("ner_tags").text.split()))  # type: ignore
                            data.append({"tokens": tokens, "ner_tags": ner_tags})
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
                        query = "SELECT tokens, ner_tags FROM dataset_table;"
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

            # Preprocess the dataset
            self.label_list = label_list if label_list else list({y for x in dataset["train"]["ner_tags"] for y in x})
            self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
            if self.model:
                config = self.model.config
                config.label2id = self.label_to_id
                config.id2label = {i: label for label, i in self.label_to_id.items()}
                config.num_labels = len(self.label_to_id.keys())
                self.config = config

                self.load_models(
                    model_name=self.model_name,
                    tokenizer_name=self.tokenizer_name,
                    model_class=self.model_class,
                    tokenizer_class=self.tokenizer_class,
                    device_map=self.device_map,
                    precision=self.precision,
                    quantization=self.quantization,
                    lora_config=self.lora_config,
                    use_accelerate=self.use_accelerate,
                    accelerate_no_split_module_classes=self.accelerate_no_split_module_classes,
                    **self.model_kwargs,
                )

                if self.tokenizer_name.lower() == "local":  # type: ignore
                    self.log.info(f"Loading local tokenizer : {self.tokenizer_class} : {self.input.get()}")
                    self.tokenizer = getattr(__import__("transformers"), str(self.tokenizer_class)).from_pretrained(
                        os.path.join(self.input.get(), "/model"),
                        add_prefix_space=True,
                    )
                else:
                    self.log.info(
                        f"Loading tokenizer from huggingface hub: {self.tokenizer_class} : {self.tokenizer_name}"
                    )
                    self.tokenizer = getattr(__import__("transformers"), str(self.tokenizer_class)).from_pretrained(
                        self.tokenizer_name,
                        revision=self.tokenizer_revision,
                        add_prefix_space=True,
                    )

                if self.tokenizer and not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id

            tokenized_dataset = dataset.map(self.prepare_train_features, batched=True)

            return tokenized_dataset
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def prepare_train_features(
        self, examples: Dict[str, Union[List[str], List[int]]]
    ) -> Dict[str, Union[List[str], List[int]]]:
        """
        Tokenize the examples and prepare the features for training.

        Args:
            examples (Dict[str, Union[List[str], List[int]]]): A dictionary of examples.

        Returns:
            Dict[str, Union[List[str], List[int]]]: The processed features.
        """

        if not self.tokenizer:
            raise Exception("Tokenizer and model have to be loaded first, please call load_models() first.")

        # convert into proper structure if coming from a csv etc
        examples["tokens"] = [
            ast.literal_eval(example) if type(example) is str else example for example in examples["tokens"]
        ]
        examples["ner_tags"] = [
            ast.literal_eval(example) if type(example) is str else example for example in examples["ner_tags"]
        ]

        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = []

        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            label_ids = []
            for word_idx in word_ids:
                if word_idx is not None:
                    self.log.debug(f"labels[word_idx]: {labels[word_idx]}")  # type: ignore
                    label_ids.append(self.label_to_id[labels[word_idx]])  # type: ignore
                else:
                    label_ids.append(-100)
            all_labels.append(label_ids)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    def data_collator(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Customize the data collator.

        Args:
            examples (List[Dict[str, torch.Tensor]]): The examples to collate.

        Returns:
            Dict[str, torch.Tensor]: The collated data.
        """
        return DataCollatorForTokenClassification(self.tokenizer)(examples)

    def compute_metrics(self, eval_prediction: EvalPrediction, average: str = "micro"):
        predictions = np.argmax(eval_prediction.predictions, axis=-1)
        labels = eval_prediction.label_ids

        # Mask out ignored values
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average=average),
            "recall": recall_score(labels, predictions, average=average),
            "f1": f1_score(labels, predictions, average=average),
        }
