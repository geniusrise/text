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
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml  # type: ignore
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from nltk.translate.bleu_score import corpus_bleu
from pyarrow import feather
from transformers import EvalPrediction

from .base import HuggingFaceFineTuner


class HuggingFaceInstructionTuningFineTuner(HuggingFaceFineTuner):
    r"""
    A bolt for fine-tuning Hugging Face models on instruction tuning tasks.

    Args:
        model: The pre-trained model to fine-tune.
        tokenizer: The tokenizer associated with the model.
        input (BatchInput): The batch input data.
        output (OutputConfig): The output data.
        state (State): The state manager.

    ## Using geniusrise to invoke via command line
    ```bash
    genius HuggingFaceInstructionTuningFineTuner rise \
        streaming \
            --input_kafka_topic webhook_test \
            --input_kafka_cluster_connection_string localhost:9094 \
            --input_kafka_consumer_group_id geniusrise \
        streaming \
            --output_kafka_topic webhook_test \
            --output_kafka_cluster_connection_string localhost:9094 \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise \
            --postgres_table state \
        listen \
            --args various=30 arguments=40 that=50 this=70 bolt=63 may=lol have='{"lol": "lel"}'
    ```

    ## Using geniusrise to invoke via YAML file
    ```yaml
    version: "1"
    bolts:
        my_instruction_bolt:
            name: "HuggingFaceInstructionTuningFineTuner"
            method: "listen"
            args:
                various: 30
                arguments: 40
                that: 50
                this: 70
                bolt: 63
                may: "lol"
                have: '{"lol": "lel"}'
            input:
                type: "streaming"
                args:
                    input_topic: "webhook_test"
                    kafka_servers: "localhost:9094"
                    group_id: "geniusrise"
            output:
                type: "streaming"
                args:
                    output_topic: "webhook_test"
                    kafka_servers: "localhost:9094"
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
                    name: "my_instruction_bolt"
                    namespace: "default"
                    image: "my_instruction_bolt_image"
                    replicas: 1
    ```
    """

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs: Any) -> Union[HFDataset, Dict]:
        r"""
        Load an instruction tuning dataset from a directory.

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
        {"instruction": "The instruction", "output": "The output"}
        ```

        ### CSV
        Should contain 'instruction' and 'output' columns.
        ```csv
        instruction,output
        "The instruction","The output"
        ```

        ### Parquet
        Should contain 'instruction' and 'output' columns.

        ### JSON
        An array of dictionaries with 'instruction' and 'output' keys.
        ```json
        [{"instruction": "The instruction", "output": "The output"}]
        ```

        ### XML
        Each 'record' element should contain 'instruction' and 'output' child elements.
        ```xml
        <record>
            <instruction>The instruction</instruction>
            <output>The output</output>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'instruction' and 'output' keys.
        ```yaml
        - instruction: "The instruction"
          output: "The output"
        ```

        ### TSV
        Should contain 'instruction' and 'output' columns separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'instruction' and 'output' columns.

        ### SQLite (.db)
        Should contain a table with 'instruction' and 'output' columns.

        ### Feather
        Should contain 'instruction' and 'output' columns.
        """

        try:
            logging.info(f"Loading dataset from {dataset_path}")
            self.max_length = max_length
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                dataset = load_from_disk(dataset_path)
                return dataset.map(self.prepare_train_features, batched=True)
            else:
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

                dataset = HFDataset.from_pandas(pd.DataFrame(data))
                return dataset.map(self.prepare_train_features, batched=True)
        except Exception as e:
            logging.error(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
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
            encoding = self.tokenizer(
                examples["instruction"],
                examples["output"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoding["labels"] = encoding["input_ids"].clone()  # Assuming that 'output' is the labels

            return encoding
        except Exception as e:
            print(f"Error preparing train features: {e}")
            raise

    def compute_metrics(self, eval_pred: EvalPrediction) -> Optional[Dict[str, float]]:
        """
        Compute evaluation metrics for the model's predictions.

        This method takes the model's predictions and ground truth labels, converts them to text,
        and then computes the BLEU score for evaluation.

        Args:
            eval_pred (EvalPrediction): A named tuple containing `predictions` and `label_ids`.
                - `predictions`: The logits predicted by the model of shape (batch_size, sequence_length, num_classes).
                - `label_ids`: The ground truth labels of shape (batch_size, sequence_length).

        Returns:
            Optional[Dict[str, float]]: A dictionary containing the BLEU score. Returns None if an exception occurs.

        Raises:
            Exception: If the tokenizer is not initialized.
        """
        predictions, labels = eval_pred
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        labels = labels[0] if isinstance(labels, tuple) else labels

        # Get the most likely token IDs from the logits (predictions)
        predictions = np.argmax(predictions, axis=1)

        # Convert labels and predictions to text
        if self.tokenizer:
            if len(labels.shape) == 1:
                labels = labels.reshape(-1, 1)
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)

            labels_text = [
                [self.tokenizer.decode(label, skip_special_tokens=True) for label in example] for example in labels
            ]
            predictions_text = [
                [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in example] for example in predictions
            ]
        else:
            raise Exception("No tokenizer found, how did we even get here, please raise a PR.")

        # Compute BLEU score
        bleu_score = corpus_bleu(labels_text, predictions_text)

        return {
            "bleu": bleu_score,
        }
