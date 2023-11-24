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

from typing import Any, Dict, List
import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise_text.base import TextAPI
import torch
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoModelForTableQuestionAnswering, AutoTokenizer


class QuestionAnsweringAPI(TextAPI):
    model: AutoModelForQuestionAnswering | AutoModelForTableQuestionAnswering
    tokenizer: AutoTokenizer

    """
    A class for handling different types of QA models: Traditional, TAPAS, and TAPEX.

    Attributes:
        model (Any): The pre-trained QA model (traditional, TAPAS, or TAPEX).
        tokenizer (Any): The tokenizer used to preprocess input text.
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs: Any,
    ):
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def answer(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Answers a question based on the provided context or table.

        Args:
            **kwargs (Any): Additional arguments to pass to the QA model.

        Returns:
            Dict[str, Any]: A dictionary containing the question, context/table, and answer.
        """
        data = cherrypy.request.json
        question = data.get("question")

        model_type = "traditional"
        if "tapas" in self.model_name.lower():
            model_type = "tapas"
        elif "tapex" in self.model_name.lower():
            model_type = "tapex"

        if model_type in ["tapas", "tapex"]:
            table = data.get("data")
            return {
                "data": table,
                "question": question,
                "answer": self.answer_table_question(table, question, model_type),
            }
        else:
            context = data.get("data")
            return {
                "data": context,
                "question": question,
                "answer": self.answer_text_question(context, question),
            }

    def answer_table_question(self, data: Dict[str, Any], question: str, model_type: str) -> dict:
        """
        Answers a question based on the provided table.

        Args:
            data (Dict[str, Any]): The table data and other parameters.
            question (str): The question to be answered.
            model_type (str): The type of the model ('tapas' or 'tapex').

        Returns:
            str: The answer derived from the table.
        """

        table = pd.DataFrame.from_dict(data)
        if model_type == "tapas":
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

            cell_answers = [self._convert_coordinates_to_answer(table, x) for x in predicted_answer_coordinates[0]]
            if type(cell_answers[0]) is list:
                cell_answers = [y for x in cell_answers for y in x]  # type: ignore

            if predicted_aggregation_indices:
                aggregation_answer = self._convert_aggregation_to_answer(predicted_aggregation_indices[0])
            else:
                aggregation_answer = "NONE"
            return {
                "answers": cell_answers,
                "aggregation": aggregation_answer,
            }

        elif model_type == "tapex":
            encoding = self.tokenizer(table, question, return_tensors="pt")
            if next(self.model.parameters()).is_cuda:
                encoding = {k: v.cuda() for k, v in encoding.items()}

            outputs = self.model.generate(**encoding)
            answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return {
                "answers": answers,
                "aggregation": "NONE",
            }
        else:
            raise ValueError("Unsupported model type for table-based QA.")

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

    def answer_text_question(self, context: str, question: str) -> dict:
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.model(**inputs)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

        answer_start = int(torch.argmax(answer_start_scores))
        answer_end = int(torch.argmax(answer_end_scores) + 1)

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )

        return {
            "answers": [answer],
            "aggregation": "NONE",
        }
