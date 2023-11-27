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
import uuid
import json
import glob
import os
import sqlite3
import xml.etree.ElementTree as ET
import pandas as pd
import yaml  # type: ignore
from datasets import Dataset, load_from_disk
from pyarrow import feather
from pyarrow import parquet as pq
import torch
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_text.base import TextBulk


class QABulk(TextBulk):
    """
    A class for bulk question-answering using Hugging Face models.
    """

    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        super().__init__(input, output, state, **kwargs)

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs) -> Optional[Dataset]:
        """
        Load a question-answering dataset from a directory.
        """

        self.max_length = max_length

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
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
                            context = record.find("data").text  # type: ignore
                            question = record.find("question").text  # type: ignore
                            data.append({"data": context, "question": question})

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
                        query = "SELECT data, question FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                return Dataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def answer_questions(
        self,
        model_name: str,
        model_class: str = "AutoModelForQuestionAnswering",
        tokenizer_class: str = "AutoTokenizer",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = True,
        batch_size: int = 32,
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
        self.batch_size = batch_size

        model_args = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}
        self.model_args = model_args

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
        dataset = self.load_dataset(dataset_path)
        if dataset is None:
            self.log.error("Failed to load dataset.")
            return

        model_type = "traditional"
        if "tapas" in self.model_name.lower():
            model_type = "tapas"
        elif "tapex" in self.model_name.lower():
            model_type = "tapex"

        output_data = []
        for batch in range(0, len(dataset), self.batch_size):
            batch_data = dataset[batch : batch + self.batch_size]

            if model_type == "traditional":
                questions = batch_data["question"]
                contexts = batch_data["data"]

                inputs = self.tokenizer(
                    questions,
                    contexts,
                    add_special_tokens=True,
                    return_tensors="pt",
                    truncation="only_second",
                    max_length=self.max_length,
                )

                # Move inputs to GPU if CUDA is available
                if self.use_cuda and torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                outputs = self.model(**inputs)

                answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
                answer_start = torch.argmax(answer_start_scores, dim=1)
                answer_end = torch.argmax(answer_end_scores, dim=1) + 1

                for i in range(outputs.start_logits.shape[0]):
                    answer = self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(
                            inputs["input_ids"][i][int(answer_start[i]) : int(answer_end[i])]
                        )
                    )
                    output_data.append(
                        {
                            "data": contexts[i],
                            "question": questions[i],
                            "answer": answer,
                        }
                    )
            elif model_type == "tapas":
                questions = batch_data["question"]
                tables = [pd.DataFrame.from_dict(json.loads(x)) for x in batch_data["data"]]

                for table, question in zip(tables, questions):
                    inputs = self.tokenizer(table=table, queries=[question], padding="max_length", return_tensors="pt")

                    if next(self.model.parameters()).is_cuda:
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    outputs = self.model(**inputs)

                    # Decode the predicted tokens
                    if hasattr(outputs, "logits_aggregation") and outputs.logits_aggregation is not None:
                        (
                            predicted_answer_coordinates,
                            predicted_aggregation_indices,
                        ) = self.tokenizer.convert_logits_to_predictions(
                            {k: v.cpu() for k, v in inputs.items()},
                            outputs.logits.detach().cpu(),
                            outputs.logits_aggregation.detach().cpu(),
                        )
                    else:
                        predicted_answer_coordinates = self.tokenizer.convert_logits_to_predictions(
                            {k: v.cpu() for k, v in inputs.items()},
                            outputs.logits.detach().cpu(),
                        )
                        predicted_aggregation_indices = None

                    cell_answers = [
                        self._convert_coordinates_to_answer(table, x) for x in predicted_answer_coordinates[0]
                    ]
                    if type(cell_answers[0]) is list:
                        cell_answers = [y for x in cell_answers for y in x]  # type: ignore

                    if predicted_aggregation_indices:
                        aggregation_answer = self._convert_aggregation_to_answer(predicted_aggregation_indices[0])
                    else:
                        aggregation_answer = "NONE"
                    output_data.append(
                        {
                            "data": table.to_dict("records"),
                            "question": question,
                            "answers": cell_answers,
                            "aggregation": aggregation_answer,
                        }
                    )

            elif model_type == "tapex":
                questions = batch_data["question"]
                tables = [pd.DataFrame.from_dict(json.loads(x)) for x in batch_data["data"]]

                for table, question in zip(tables, questions):
                    encoding = self.tokenizer(table, question, return_tensors="pt")
                    if next(self.model.parameters()).is_cuda:
                        encoding = {k: v.cuda() for k, v in encoding.items()}

                    outputs = self.model.generate(**encoding)
                    answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    output_data.append(
                        {
                            "data": table.to_dict("records"),
                            "question": question,
                            "answers": answers,
                            "aggregation": "NONE",
                        }
                    )

        # Save the results
        output_file = os.path.join(output_path, f"qa_results-{str(uuid.uuid4())}.json")
        with open(output_file, "w") as file:
            json.dump(output_data, file)
        self.log.info(f"Results saved to {output_file}")

    def _convert_aggregation_to_answer(self, aggregation_index: int) -> str:
        """
        Converts the aggregation index predicted by TAPAS into an aggregation operation.

        Args:
            aggregation_index (int): The index of the aggregation operation.

        Returns:
            str: The string representation of the aggregation operation.
        """
        aggregation_operations = {
            0: "NONE",
            1: "SUM",
            2: "AVERAGE",
            3: "COUNT",
            4: "MIN",
            5: "MAX",
            6: "OR",
            7: "AND",
            8: "CONCAT",
            9: "FIRST",
            10: "LAST",
        }
        return aggregation_operations.get(aggregation_index, "NONE")

    def _convert_coordinates_to_answer(self, table: pd.DataFrame, coordinates: Any) -> List[str]:
        """
        Converts the coordinates predicted by TAPAS into an answer string.

        Args:
            table (pd.DataFrame): The table used for the QA.
            coordinates (Any): The coordinates of the cells predicted as part of the answer.

        Returns:
            List[str]: The answer strings.
        """
        if type(coordinates) is tuple:
            coordinates = [coordinates]
        return [table.iat[coord] for coord in coordinates]
