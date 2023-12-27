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
    r"""
    TextClassificationAPI is a text classification service leveraging Hugging Face's transformers to provide an API
    for text classification tasks. It supports various models for sequence classification tasks, including sentiment
    analysis, topic classification, intent recognition, etc.

    Attributes:
        model (AutoModelForSequenceClassification): A Hugging Face model for sequence classification tasks.
        tokenizer (AutoTokenizer): A tokenizer that preprocesses text for the model.

    Example CLI Usage:
    ```bash
    genius TextClassificationAPI rise \
        batch \
            --input_s3_bucket geniusrise-test \
            --input_s3_folder none \
        batch \
            --output_s3_bucket geniusrise-test \
            --output_s3_folder none \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise\
            --postgres_table state \
        --id cardiffnlp/twitter-roberta-base-hate-multiclass-latest-lol \
        listen \
            --args \
                model_name="cardiffnlp/twitter-roberta-base-hate-multiclass-latest" \
                model_class="AutoModelForSequenceClassification" \
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
        **kwargs,
    ) -> None:
        """
        Initializes the TextClassificationAPI with the necessary configurations for input, output, and state management.

        Args:
            input (BatchInput): Configuration for the input data.
            output (BatchOutput): Configuration for the output data.
            state (State): State management for the API.
            **kwargs: Additional keyword arguments for extended functionality.
        """
        super().__init__(input=input, output=output, state=state)
        log.info("Loading Hugging Face API server")

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def classify(self) -> Dict[str, Any]:
        """
        Accepts text input and returns classification results. The method uses the model and tokenizer to classify the text
        and provide the likelihood of each class label.

        Returns:
            Dict[str, Any]: A dictionary containing the original input text and the classification scores for each label.

        Example CURL Request for text classification:
        ```bash
        /usr/bin/curl -X POST localhost:3000/api/v1/classify \
            -H "Content-Type: application/json" \
            -d '{
                "text": "tata sons lost a major contract to its rival mahindra motors"
            }' | jq
        ```
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
        label_scores = {id_to_label[label_id]: score for label_id, score in enumerate(scores[0])}

        return {"input": text, "label_scores": label_scores}
