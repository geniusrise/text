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
from typing import Any, Dict
import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise_text.base import TextAPI

log = logging.getLogger(__name__)


class SummarizationAPI(TextAPI):
    """
    A class for serving a Hugging Face-based summarization model.

    Attributes:
        model (AutoModelForSeq2SeqLM): The loaded Hugging Face model for summarization.
        tokenizer (AutoTokenizer): The tokenizer for preprocessing text.

    Methods:
        summarize(self, **kwargs: Any) -> Dict[str, Any]:
            Summarizes the input text based on the given parameters.
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SummarizationAPI class.

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
    def summarize(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Summarizes the input text based on the given parameters.

        Args:
            **kwargs (Any): Additional arguments for summarization.

        Returns:
            Dict[str, Any]: A dictionary containing the input text and its summary.
        """
        data = cherrypy.request.json
        text = data.get("text")
        decoding_strategy = data.get("decoding_strategy", "generate")

        generation_params = data
        if "decoding_strategy" in generation_params:
            del generation_params["decoding_strategy"]

        summary = self.generate(prompt=text, decoding_strategy=decoding_strategy, **generation_params)

        return {"input": text, "summary": summary}
