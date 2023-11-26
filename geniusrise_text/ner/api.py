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

from typing import Any, Dict
import torch
import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_text.base import TextAPI
from geniusrise.logging import setup_logger


class NamedEntityRecognitionAPI(TextAPI):
    """
    A class for serving a Hugging Face-based Named Entity Recognition (NER) model.

    Attributes:
        model (Any): The loaded NER model.
        tokenizer (Any): The tokenizer for preprocessing text.

    Methods:
        recognize_entities(self, **kwargs: Any) -> Dict[str, Any]:
            Recognizes named entities in the input text based on the given parameters.
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the NamedEntityRecognitionAPI class.

        Args:
            input (BatchInput): The input data.
            output (BatchOutput): The output data.
            state (State): The state data.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def recognize_entities(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Recognizes named entities in the input text based on the given parameters.

        Args:
            **kwargs (Any): Additional arguments for entity recognition.

        Returns:
            Dict[str, Any]: A dictionary containing the input text and its recognized entities.
        """
        data = cherrypy.request.json
        text = data.get("text")
        generation_args = data

        if "text" in generation_args:
            del generation_args["text"]

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, **generation_args)
            predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

        entities = [
            {"token": self.tokenizer.convert_ids_to_tokens(i), "class": self.model.config.id2label[x]}
            for (x, i) in zip(predictions, inputs["input_ids"].squeeze().tolist())
        ]

        return {"input": text, "entities": entities}
