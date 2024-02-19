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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from geniusrise_text.base import TextAPI

log = logging.getLogger(__name__)


class SummarizationAPI(TextAPI):
    r"""
    A class for serving a Hugging Face-based summarization model. This API provides an interface to
    submit text and receive a summarized version, utilizing state-of-the-art machine learning models for
    text summarization.

    Attributes:
        model (AutoModelForSeq2SeqLM): The loaded Hugging Face model for summarization.
        tokenizer (AutoTokenizer): The tokenizer for preprocessing text.

    Methods:
        summarize(self, **kwargs: Any) -> Dict[str, Any]:
            Summarizes the input text based on the given parameters.

    CLI Usage:
    ```bash
    genius SummarizationAPI rise \
        batch \
            --input_folder ./input \
        batch \
            --output_folder ./output \
        none \
        --id facebook/bart-large-cnn-lol \
        listen \
            --args \
                model_name="facebook/bart-large-cnn" \
                model_class="AutoModelForSeq2SeqLM" \
                tokenizer_class="AutoTokenizer" \
                use_cuda=True \
                precision="float" \
                quantization=0 \
                device_map="cuda:0" \
                max_memory=None \
                torchscript=False \
                endpoint="*" \
                port=3000 \
                cors_domain="http://localhost:3000" \
                username="user" \
                password="password"
    ```
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the SummarizationAPI class with input, output, and state configurations.

        Args:
            input (BatchInput): Configuration for input data.
            output (BatchOutput): Configuration for output data.
            state (State): State management for API.
            **kwargs (Any): Additional keyword arguments for extended functionality.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)
        self.pipeline = None

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def summarize(self, **kwargs: Any) -> Dict[str, Any]:
        r"""
        Summarizes the input text based on the given parameters using a machine learning model. The method
        accepts parameters via a POST request and returns the summarized text.

        Args:
            **kwargs (Any): Arbitrary keyword arguments. Expected to receive these from the POST request's JSON body.

        Returns:
            Dict[str, Any]: A dictionary containing the input text and its summary.

        Example CURL Requests:
        ```bash
        /usr/bin/curl -X POST localhost:3000/api/v1/summarize \
            -H "Content-Type: application/json" \
            -d '{
                "text": "Theres something magical about Recurrent Neural Networks (RNNs). I still remember when I trained my first recurrent network for Image Captioning. Within a few dozen minutes of training my first baby model (with rather arbitrarily-chosen hyperparameters) started to generate very nice looking descriptions of images that were on the edge of making sense. Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times. What made this result so shocking at the time was that the common wisdom was that RNNs were supposed to be difficult to train (with more experience Ive in fact reached the opposite conclusion). Fast forward about a year: Im training RNNs all the time and Ive witnessed their power and robustness many times, and yet their magical outputs still find ways of amusing me.",
                "decoding_strategy": "generate",
                "bos_token_id": 0,
                "decoder_start_token_id": 2,
                "early_stopping": true,
                "eos_token_id": 2,
                "forced_bos_token_id": 0,
                "forced_eos_token_id": 2,
                "length_penalty": 2.0,
                "max_length": 142,
                "min_length": 56,
                "no_repeat_ngram_size": 3,
                "num_beams": 4,
                "pad_token_id": 1,
                "do_sample": false
            }' | jq
        ```

        ```bash
        /usr/bin/curl -X POST localhost:3000/api/v1/summarize \
            -H "Content-Type: application/json" \
            -d '{
                "text": "Theres something magical about Recurrent Neural Networks (RNNs). I still remember when I trained my first recurrent network for Image Captioning. Within a few dozen minutes of training my first baby model (with rather arbitrarily-chosen hyperparameters) started to generate very nice looking descriptions of images that were on the edge of making sense. Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times. What made this result so shocking at the time was that the common wisdom was that RNNs were supposed to be difficult to train (with more experience Ive in fact reached the opposite conclusion). Fast forward about a year: Im training RNNs all the time and Ive witnessed their power and robustness many times, and yet their magical outputs still find ways of amusing me.",
                "decoding_strategy": "generate",
                "early_stopping": true,
                "length_penalty": 2.0,
                "max_length": 142,
                "min_length": 56,
                "no_repeat_ngram_size": 3,
                "num_beams": 4
            }' | jq
        ```
        """
        data = cherrypy.request.json
        text = data.get("text")
        decoding_strategy = data.get("decoding_strategy", "generate")

        generation_params = data
        if "decoding_strategy" in generation_params:
            del generation_params["decoding_strategy"]

        summary = self.generate(prompt=text, decoding_strategy=decoding_strategy, **generation_params)

        return {"input": text, "summary": summary}

    def initialize_pipeline(self):
        """
        Lazy initialization of the summarization Hugging Face pipeline.
        """
        if not self.hf_pipeline:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.hf_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def summarize_pipeline(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Summarizes the input text using the Hugging Face pipeline based on given parameters.

        Args:
            **kwargs: Keyword arguments containing parameters for summarization.

        Returns:
            A dictionary containing the input text and its summary.

        Example CURL Request for summarization:
        `curl -X POST localhost:3000/api/v1/summarize_pipeline -H "Content-Type: application/json" -d '{"text": "Your long text here"}'`
        """
        self.initialize_pipeline()  # Initialize the pipeline on first API hit

        data = cherrypy.request.json
        text = data.get("text")
        generation_params = {k: v for k, v in data.items() if k != "text"}

        result = self.hf_pipeline(text, **generation_params)  # type: ignore

        return {"input": text, "summary": result}
