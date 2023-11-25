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
from geniusrise_text.base import TextAPI

log = logging.getLogger(__name__)


class TranslationAPI(TextAPI):
    """
    A class for serving a Hugging Face-based translation model.
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs: Any,
    ) -> None:
        super().__init__(input=input, output=output, state=state)
        log.info("Loading Hugging Face translation API server")

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def translate(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Translates text to a specified target language.
        """
        data = cherrypy.request.json
        text = data.get("text")
        decoding_strategy = data.get("decoding_strategy", "generate")
        src_lang = data.get("source_lang")
        target_lang = data.get("target_lang", "en")

        generation_params = data
        if "decoding_strategy" in generation_params:
            del generation_params["decoding_strategy"]
        if "source_lang" in generation_params:
            del generation_params["source_lang"]
        if "target_lang" in generation_params:
            del generation_params["target_lang"]
        if "text" in generation_params:
            del generation_params["text"]

        # Tokenize the text
        self.tokenizer.src_lang = src_lang
        if target_lang != "en":
            generation_params = {
                **generation_params,
                **{"forced_bos_token_id": self.tokenizer.lang_code_to_id[target_lang]},
            }

        translated_text = self.generate(prompt=text, decoding_strategy=decoding_strategy, **generation_params)

        return {
            "text": text,
            "target_language": src_lang,
            "translated_text": translated_text,
        }
