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

import ast
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import pandas as pd
import yaml  # type: ignore
from datasets import Dataset, load_from_disk
from geniusrise.core import BatchInput, BatchOutput, State
from pyarrow import feather
from pyarrow import parquet as pq
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction

from geniusrise_huggingface.base import HuggingFaceFineTuner


class HuggingFaceQuestionAnsweringFineTuner(HuggingFaceFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on question answering tasks.

    Args:
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    ## Using geniusrise to invoke via command line
    ```bash
    genius HuggingFaceQuestionAnsweringFineTuner rise \
        batch \
            --input_bucket my_bucket \
            --input_folder my_folder \
        batch \
            --output_bucket my_output_bucket \
            --output_folder my_output_folder \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise \
            --postgres_table state \
        load_dataset \
            --args dataset_path=my_dataset_path max_length=512
    ```

    ## Using geniusrise to invoke via YAML file
    ```yaml
    version: "1"
    bolts:
        my_huggingface_bolt:
            name: "HuggingFaceQuestionAnsweringFineTuner"
            method: "load_dataset"
            args:
                dataset_path: "my_dataset_path"
                max_length: 512
            input:
                type: "batch"
                args:
                    bucket: "my_bucket"
                    folder: "my_folder"
            output:
                type: "batch"
                args:
                    bucket: "my_output_bucket"
                    folder: "my_output_folder"
            state:
                type: "postgres"
                args:
                    postgres_host: "127.0.0.1"
                    postgres_port: 5432
                    postgres_user: "postgres"
                    postgres_password: "postgres"
                    postgres_database: "geniusrise"
                    postgres_table: "state"
            deploy:
                type: "k8s"
                args:
                    name: "my_huggingface_bolt"
                    namespace: "default"
                    image: "my_huggingface_bolt_image"
                    replicas: 1
    ```
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs: Dict[str, Any],
    ) -> None:
        r"""
        Initialize the bolt.

        ```
        Args:
            input (BatchInput): The batch input data.
            output (OutputConfig): The output data.
            state (State): The state manager.
            **kwargs: Additional keyword arguments.
        ```
        """
        self.pad_on_right = True
        self.max_length = 0
        self.doc_stride = 0
        self.evaluate_squadv2 = False

        super().__init__(
            input=input,
            output=output,
            state=state,
            eval=eval,
            **kwargs,
        )

    def load_dataset(
        self,
        dataset_path: str,
        pad_on_right: bool = True,
        max_length: Optional[int] = None,
        doc_stride: Optional[int] = None,
        evaluate_squadv2: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Optional[Dataset]:
        r"""
        Load a dataset from a directory.

        ## Supported Data Formats and Structures:

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"context": "The context content", "question": "The question", "answers": {"answer_start": [int], "text": [str]}}
        ```

        ### CSV
        Should contain 'context', 'question', and 'answers' columns.
        ```csv
        context,question,answers
        "The context content","The question","{'answer_start': [int], 'text': [str]}"
        ```

        ### Parquet
        Should contain 'context', 'question', and 'answers' columns.

        ### JSON
        An array of dictionaries with 'context', 'question', and 'answers' keys.
        ```json
        [{"context": "The context content", "question": "The question", "answers": {"answer_start": [int], "text": [str]}}]
        ```

        ### XML
        Each 'record' element should contain 'context', 'question', and 'answers' child elements.
        ```xml
        <record>
            <context>The context content</context>
            <question>The question</question>
            <answers answer_start="int" text="str"></answers>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'context', 'question', and 'answers' keys.
        ```yaml
        - context: "The context content"
          question: "The question"
          answers:
            answer_start: [int]
            text: [str]
        ```

        ### TSV
        Should contain 'context', 'question', and 'answers' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'context', 'question', and 'answers' columns.

        ### SQLite (.db)
        Should contain a table with 'context', 'question', and 'answers' columns.

        ### Feather
        Should contain 'context', 'question', and 'answers' columns.

        Args:
            dataset_path (str): The path to the dataset directory.
            pad_on_right (bool): Whether to pad on the right.
            max_length (int): The maximum length of the sequences.
            doc_stride (int): The document stride.
            evaluate_squadv2 (bool): Whether to evaluate using SQuAD v2 metrics.

        Returns:
            Dataset: The loaded dataset.
        """

        # Update padding, max_length, and doc_stride if provided
        self.pad_on_right = pad_on_right
        self.max_length = max_length if max_length is not None else self.max_length
        self.doc_stride = doc_stride if doc_stride is not None else self.doc_stride
        self.evaluate_squadv2 = evaluate_squadv2

        # Load the dataset from the directory
        try:
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
                            context = record.find("context").text  # type: ignore
                            question = record.find("question").text  # type: ignore
                            answers = record.find("answers").text  # type: ignore
                            data.append(
                                {
                                    "context": context,
                                    "question": question,
                                    "answers": answers,
                                }
                            )
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
                        query = "SELECT context, question, answers FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))
                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))
                dataset = Dataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            return None

        # Preprocess the dataset
        try:
            tokenized_dataset = dataset.map(
                self.prepare_train_features,
                batched=True,
                remove_columns=dataset.column_names,
            )
        except Exception as e:
            self.log.exception(f"Error tokenizing dataset: {e}")
            raise

        return tokenized_dataset

    def prepare_train_features(
        self, examples: Dict[str, Union[str, List[str]]]
    ) -> Optional[Dict[str, Union[List[int], List[List[int]]]]]:
        """
        Tokenize our examples with truncation and padding, but keep the overflows using a stride.

        Args:
            examples: The examples to be tokenized.

        Returns:
            The tokenized examples.
        """
        if not self.tokenizer:
            raise Exception("No tokenizer found, please call load_models first.")

        examples["answers"] = [
            ast.literal_eval(example) if type(example) is str else example for example in examples["answers"]
        ]

        # Tokenize the examples
        try:
            tokenized_examples = self.tokenizer(
                examples["question" if self.pad_on_right else "context"],
                examples["context" if self.pad_on_right else "question"],
                truncation="only_second" if self.pad_on_right else "only_first",
                max_length=self.max_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
        except Exception as e:
            self.log.exception(f"Error tokenizing examples: {e}")
            raise

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index][0]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:  # type: ignore
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]  # type: ignore
                end_char = start_char + len(answers["text"][0])  # type: ignore

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the
                    # last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def compute_metrics(self, eval_pred: EvalPrediction) -> Optional[Dict[str, float]]:
        """
        Compute the accuracy of the model's predictions.

        Args:
            eval_pred (tuple): A tuple containing two elements:
                - predictions (np.ndarray): The model's predictions.
                - label_ids (np.ndarray): The true labels.

        Returns:
            dict: A dictionary mapping metric names to computed values.
        """
        # Compute the metrics
        try:
            predictions, labels = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            if isinstance(labels, tuple):
                labels = labels[0]

            # Convert predictions from list of 1D arrays to 1D array
            predictions = np.array([np.argmax(p) for p in predictions])

            if self.evaluate_squadv2:
                metric = evaluate.load("squad_v2" if self.evaluate_squadv2 else "squad")
                squad = metric.compute(predictions=predictions, references=eval_pred.label_ids)
            else:
                squad = {}

            return {"accuracy": accuracy_score(labels, predictions)}
        except Exception as e:
            self.log.exception(f"Error computing metrics: {e}")
            raise
