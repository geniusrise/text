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
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union

import pandas as pd
import pyarrow.parquet as pq
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from pyarrow import feather
from transformers import DataCollatorWithPadding

from geniusrise_text.base import TextFineTuner


class NLIFineTuner(TextFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models for text classification tasks.

    This class extends the `TextFineTuner` and specializes in fine-tuning models for text classification.
    It provides additional functionalities for loading and preprocessing text classification datasets in various formats.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    CLI Usage:

    ```bash
        genius NLIFineTuner rise \
            batch \
                --input_folder ./input \
            batch \
                --output_folder ./output \
            none \
            --id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-lol
            fine_tune \
                --args \
                    model_name=my_model \
                    tokenizer_name=my_tokenizer \
                    num_train_epochs=3 \
                    per_device_train_batch_size=8
    ```
    """

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> Union[Dataset, DatasetDict, None]:
        r"""
        Load a commonsense reasoning dataset from a directory.

        Args:
            dataset_path (str): The path to the dataset directory.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### Hugging Face Dataset
        Dataset files saved by the Hugging Face datasets library.

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"premise": "The premise text", "hypothesis": "The hypothesis text", "label": 0 or 1 or 2}
        ```

        ### CSV
        Should contain 'premise', 'hypothesis', and 'label' columns.
        ```csv
        premise,hypothesis,label
        "The premise text","The hypothesis text",0
        ```

        ### Parquet
        Should contain 'premise', 'hypothesis', and 'label' columns.

        ### JSON
        An array of dictionaries with 'premise', 'hypothesis', and 'label' keys.
        ```json
        [{"premise": "The premise text", "hypothesis": "The hypothesis text", "label": 0}]
        ```

        ### XML
        Each 'record' element should contain 'premise', 'hypothesis', and 'label' child elements.
        ```xml
        <record>
            <premise>The premise text</premise>
            <hypothesis>The hypothesis text</hypothesis>
            <label>0</label>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'premise', 'hypothesis', and 'label' keys.
        ```yaml
        - premise: "The premise text"
          hypothesis: "The hypothesis text"
          label: 0
        ```

        ### TSV
        Should contain 'premise', 'hypothesis', and 'label' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'premise', 'hypothesis', and 'label' columns.

        ### SQLite (.db)
        Should contain a table with 'premise', 'hypothesis', and 'label' columns.

        ### Feather
        Should contain 'premise', 'hypothesis', and 'label' columns.
        """

        try:
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
                            data.extend(json.load(f))
                    elif filename.endswith(".xml"):
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for record in root.findall("record"):
                            example = {
                                "premise": record.find("premise").text,  # type: ignore
                                "hypothesis": record.find("hypothesis").text,  # type: ignore
                                "label": int(record.find("label").text),  # type: ignore
                            }
                            data.append(example)
                    elif filename.endswith((".yaml", ".yml")):
                        with open(filepath, "r") as f:
                            data.extend(yaml.safe_load(f))
                    elif filename.endswith(".tsv"):
                        df = pd.read_csv(filepath, sep="\t")
                        data.extend(df.to_dict("records"))
                    elif filename.endswith((".xls", ".xlsx")):
                        df = pd.read_excel(filepath)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".db"):
                        conn = sqlite3.connect(filepath)
                        query = "SELECT premise, hypothesis, label FROM dataset_table;"
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

            # Create label_to_id mapping and save it in model config
            # TODO: ugly shit cause we dont know num labels before we process the data but need tokenizer to process data
            self.label_to_id = {label: i for i, label in enumerate(set(dataset["train"]["label"]))}
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
            if self.tokenizer and not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

            return dataset.map(self.prepare_train_features, batched=True)

        except Exception as e:
            self.log.exception(f"Error loading dataset: {e}")
            raise

    def prepare_train_features(self, examples: Dict) -> Dict:
        """
        Tokenize the examples and prepare the features for training.

        Args:
            examples (dict): A dictionary of examples.

        Returns:
            dict: The processed features.
        """
        try:
            if not self.tokenizer:
                raise Exception("Tokenizer not initialized")

            # Tokenize the examples
            tokenized_inputs = self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding=True,
            )

            # Prepare the labels
            tokenized_inputs["labels"] = examples["label"]

            return tokenized_inputs
        except Exception as e:
            self.log.exception(f"Error preparing train features: {e}")
            raise

    def data_collator(self, examples: Dict) -> Dict:
        """
        Customize the data collator.

        Args:
            examples: The examples to collate.

        Returns:
            dict: The collated data.
        """
        try:
            return DataCollatorWithPadding(self.tokenizer)(examples)

        except Exception as e:
            self.log.exception(f"Error in data collation: {e}")
            raise
