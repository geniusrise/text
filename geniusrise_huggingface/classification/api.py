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
from typing import Dict

import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from geniusrise_huggingface.base import HuggingFaceAPI

log = logging.getLogger(__file__)


class HuggingFaceClassificationAPI(HuggingFaceAPI):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ) -> None:
        super().__init__(input=input, output=output, state=state)
        log.info("Loading Hugging Face API server")

    def load_models(self, model_path: str, tokenizer_path: str) -> None:
        """
        Load the model and tokenizer.

        Args:
            model_path (str): The path to the saved model.
            tokenizer_path (str): The path to the saved tokenizer.
        """
        log.info(f"Loading model from {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        log.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def classify(self) -> Dict[str, str]:
        """
        Classify the input text.

        Returns:
            Dict[str, str]: The classification result.
        """
        data = cherrypy.request.json
        text = data.get("text", "")
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        label_id = outputs.logits.argmax(-1).item()
        label = self.model.config.id2label[label_id]
        return {"label": label}
