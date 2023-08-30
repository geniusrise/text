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
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq
import torch
import yaml  # type: ignore
from datasets import Dataset, DatasetDict, load_from_disk
from geniusrise.core import BatchInput, BatchOutput, State
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    DataCollatorForTokenClassification,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .base import HuggingFaceFineTuner


class HuggingFaceNamedEntityRecognitionFineTuner(HuggingFaceFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on named entity recognition tasks.

    ```
    Args:
        model: The pre-trained model to fine-tune.
        tokenizer: The tokenizer associated with the model.
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.
    ```
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        label_list: List[str],
        **kwargs,
    ):
        r"""
        Initialize the NamedEntityRecognitionFineTuner.

        ```
        Args:
            model: The pre-trained model to fine-tune.
            tokenizer: The tokenizer associated with the model.
            input (BatchInput): The batch input data.
            output (BatchOutput): The batch output data.
            state (State): The state manager.
            label_list (List[str]): The list of labels for the NER task.
            **kwargs: Additional arguments for the superclass.
        ```
        """
        self.label_list = label_list
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            input=input,
            output=output,
            state=state,
            **kwargs,
        )

    def load_dataset(self, dataset_path: str, **kwargs: Any) -> DatasetDict:
        r"""
        Load a named entity recognition dataset from a directory.

        ```
        The directory can contain any of the following file types:
        - Dataset files saved by the Hugging Face datasets library.
        - JSONL files: Each line is a JSON object representing an example. Structure:
            {
                "tokens": ["token1", "token2", ...],
                "ner_tags": [0, 1, ...]
            }
        - CSV files: Should contain 'tokens' and 'ner_tags' columns.
        - Parquet files: Should contain 'tokens' and 'ner_tags' columns.
        - JSON files: Should be an array of objects with 'tokens' and 'ner_tags' keys.
        - XML files: Each 'record' element should contain 'tokens' and 'ner_tags' child elements.
        - YAML/YML files: Each document should be a dictionary with 'tokens' and 'ner_tags' keys.
        - TSV files: Should contain 'tokens' and 'ner_tags' columns separated by tabs.
        - Excel files (.xls, .xlsx): Should contain 'tokens' and 'ner_tags' columns.
        - SQLite files (.db): Should contain a table with 'tokens' and 'ner_tags' columns.
        - Feather files: Should contain 'tokens' and 'ner_tags' columns.
        ```

        Args:
            dataset_path (str): The path to the dataset directory.

        Returns:
            DatasetDict: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.
        """
        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
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

            # Preprocess the dataset
            tokenized_dataset = dataset.map(
                self.prepare_train_features,
                batched=True,
                remove_columns=dataset.column_names,
            )

            return tokenized_dataset
        except Exception as e:
            self.log.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
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
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):  # assuming the key in your examples dict is 'ner_tags'
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                # assign label of the word to all subwords
                if word_idx is not None:
                    label_ids.append(self.label_to_id[label[word_idx]])  # type: ignore
                else:
                    label_ids.append(-100)
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
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
