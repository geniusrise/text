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

import numpy as np
import torch
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from geniusrise import BatchInput, BatchOutput, State, StreamingInput, StreamingOutput
from geniusrise_text.base import TextBulk, TextStream


class _TextClassificationInference:
    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer

    def classify_text(self, text: str, generation_params: dict):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, **generation_params)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if next(self.model.parameters()).is_cuda:
                logits = logits.cpu()

            # Handling a single number output
            if logits.numel() == 1:
                logits = outputs.logits.cpu().detach().numpy()
                scores = 1 / (1 + np.exp(-logits)).flatten()
                return {"input": text, "label_scores": scores.tolist()}
            else:
                softmax = torch.nn.functional.softmax(logits, dim=-1)
                scores = softmax.numpy().tolist()

        id_to_label = dict(enumerate(self.model.config.id2label.values()))  # type: ignore
        label_scores = {id_to_label[label_id]: score for label_id, score in enumerate(scores[0])}

        return {"input": text, "label_scores": label_scores}

    def classify_text_batch(self, batch: List[str], generation_params: dict):
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

        if next(self.model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        predictions = self.model(**inputs, **generation_params)
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions.logits
        predictions = torch.argmax(predictions, dim=-1).cpu().numpy()
        return predictions


class TextClassificationInference(TextBulk, _TextClassificationInference):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the TextToSpeechAPI with configurations for text-to-speech processing.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)


class TextClassificationInferenceStream(TextStream, _TextClassificationInference):
    def __init__(
        self,
        input: StreamingInput,
        output: StreamingOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the SpeechToTextAPI with configurations for speech-to-text processing.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)
