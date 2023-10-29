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

import logging
from typing import Dict, Any
import torch
import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_text.base import TextAPI

log = logging.getLogger(__file__)


class TextClassificationAPI(TextAPI):
    """
    A class for serving a Hugging Face-based classification model.

    Args:
        input (BatchInput): The input data.
        output (BatchOutput): The output data.
        state (State): The state data.
        **kwargs: Additional keyword arguments.

    Attributes:
        model (AutoModelForSequenceClassification): The loaded Hugging Face model.
        tokenizer (AutoTokenizer): The loaded Hugging Face tokenizer.
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ) -> None:
        """
        Initializes the TextClassificationAPI class.

        Args:
            input (BatchInput): The input data.
            output (BatchOutput): The output data.
            state (State): The state data.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state)
        log.info("Loading Hugging Face API server")

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def classify(self) -> Dict[str, Any]:
        """
        Classify the input text.

        Returns:
            Dict[str, str]: The classification result.
        """
        data: Dict[str, str] = cherrypy.request.json
        text = data.get("text", "")

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

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
        label_scores = [{id_to_label[label_id]: score} for label_id, score in enumerate(scores[0])]

        return {"input": text, "label_scores": label_scores}
