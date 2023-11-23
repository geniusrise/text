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
import torch
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise_text.base import TextAPI


class NLIAPI(TextAPI):
    """
    A class representing a Hugging Face API for Natural Language Inference (NLI).

    Args:
        input (BatchInput): The input data.
        output (BatchOutput): The output data.
        state (State): The state data.
        **kwargs: Additional keyword arguments.

    Attributes:
        model (AutoModelForSequenceClassification): The loaded Hugging Face model.
        tokenizer (AutoTokenizer): The loaded Hugging Face tokenizer.
    """

    model: Any
    tokenizer: Any

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs: Any,
    ):
        """
        Initializes a new instance of the NLIAPI class.

        Args:
            input (BatchInput): The input data to process.
            output (BatchOutput): The output data to process.
            state (State): The state of the API.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def entailment(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Evaluates the relationship between the premise and hypothesis.

        Returns:
            Dict[str, Any]: A dictionary containing the premise, hypothesis, and their relationship scores.
        """
        data = cherrypy.request.json
        premise = data.get("premise", "")
        hypothesis = data.get("hypothesis", "The statement is true")

        inputs = self.tokenizer(
            premise,
            hypothesis,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        print(inputs)
        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if next(self.model.parameters()).is_cuda:
                logits = logits.cpu()
            softmax = torch.nn.functional.softmax(logits, dim=-1)
            scores = softmax.numpy().tolist()  # Convert scores to list

        id_to_label = dict(enumerate(self.model.config.id2label.values()))  # type: ignore
        label_scores = {id_to_label[label_id]: score for label_id, score in enumerate(scores[0])}

        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "label_scores": label_scores,
        }

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def classify(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Classifies the input text into one of the candidate labels using zero-shot classification.

        Args:
            text (str): The input text to classify.
            candidate_labels (List[str], optional): The list of candidate labels to choose from.

        Returns:
            Dict[str, Any]: A dictionary containing the input text, candidate labels, and classification scores.
        """
        data = cherrypy.request.json
        text = data.get("text", "")
        candidate_labels = data.get("candidate_labels", [])

        label_scores = {}
        for label in candidate_labels:
            # Construct hypothesis for each label
            hypothesis = f"This example is {label}."

            # Tokenize the text and hypothesis
            inputs = self.tokenizer(text, hypothesis, return_tensors="pt", padding=True, truncation=True)

            # Move inputs to GPU if CUDA is enabled
            if self.use_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                softmax = torch.nn.functional.softmax(logits, dim=-1)
                scores = softmax.cpu().numpy().tolist()

            # Consider 'entailment' score as the label score
            entailment_idx = self.model.config.label2id.get("entailment", 0)
            label_scores[label] = scores[0][entailment_idx]

        sum_scores = sum(label_scores.values())
        label_scores = {k: v / sum_scores for k, v in label_scores.items()}
        return {"text": text, "label_scores": label_scores}

    def _get_entailment_scores(self, premise: str, hypotheses: List[str]) -> Dict[str, float]:
        """
        Helper method to get entailment scores for multiple hypotheses.

        Args:
            premise (str): The input premise text.
            hypotheses (List[str]): A list of hypothesis texts.

        Returns:
            Dict[str, float]: A dictionary mapping each hypothesis to its entailment score.
        """
        label_scores = {}
        for hypothesis in hypotheses:
            inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)
            if self.use_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                softmax = torch.nn.functional.softmax(logits, dim=-1)
                scores = softmax.cpu().numpy().tolist()

            entailment_idx = self.model.config.label2id.get("entailment", 0)
            label_scores[hypothesis] = scores[0][entailment_idx]

        return label_scores

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def textual_similarity(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Evaluates the textual similarity between two texts.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            Dict[str, Any]: A dictionary containing similarity score.
        """
        data = cherrypy.request.json
        text1 = data.get("text1", "")
        text2 = data.get("text2", "")

        # Using the same text as premise and hypothesis for similarity
        scores = self._get_entailment_scores(text1, [text2])
        return {"text1": text1, "text2": text2, "similarity_score": scores[text2]}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def fact_checking(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Performs fact checking on a statement given a context.

        Args:
            context (str): The context or background information.
            statement (str): The statement to fact check.

        Returns:
            Dict[str, Any]: A dictionary containing fact checking scores.
        """
        data = cherrypy.request.json
        context = data.get("context", "")
        statement = data.get("statement", "")

        scores = self._get_entailment_scores(context, [statement])
        return {
            "context": context,
            "statement": statement,
            "fact_checking_score": scores[statement],
        }

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def question_answering(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Performs question answering for multiple choice questions.

        Args:
            question (str): The question text.
            choices (List[str]): A list of possible answers.

        Returns:
            Dict[str, Any]: A dictionary containing the scores for each answer choice.
        """
        data = cherrypy.request.json
        question = data.get("question", "")
        choices = data.get("choices", [])

        scores = self._get_entailment_scores(question, choices)
        return {"question": question, "choices": choices, "scores": scores}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def detect_intent(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Detects the intent of the input text from a list of possible intents.

        Args:
            text (str): The input text.
            intents (List[str]): A list of possible intents.

        Returns:
            Dict[str, Any]: A dictionary containing the input text and detected intent with its score.
        """
        data = cherrypy.request.json
        text = data.get("text", "")
        intents = data.get("intents", [])

        # Zero-shot classification for intent detection
        scores = self._get_entailment_scores(text, intents)
        return {"text": text, "intents": intents, "scores": scores}
