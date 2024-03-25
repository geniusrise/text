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

import asyncio
from typing import Any, Optional

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

from geniusrise import BatchInput, BatchOutput, State, StreamingInput, StreamingOutput
from geniusrise_text.base import TextBulk, TextStream


class _InstructionInference:
    model: Any
    tokenizer: Any
    vllm_server: Optional[OpenAIServingChat] = None

    def infer_vllm(self, model_name: str, data: dict):
        # Prepare the chat completion request
        chat_request = ChatCompletionRequest(
            model=model_name,
            messages=data.get("messages"),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 1.0),
            n=data.get("n", 1),
            max_tokens=data.get("max_tokens"),
            stop=data.get("stop", []),
            # stream=data.get("stream", False),
            presence_penalty=data.get("presence_penalty", 0.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            logit_bias=data.get("logit_bias", {}),
            user=data.get("user"),
            best_of=data.get("best_of"),
            top_k=data.get("top_k", -1),
            ignore_eos=data.get("ignore_eos", False),
            use_beam_search=data.get("use_beam_search", False),
            stop_token_ids=data.get("stop_token_ids", []),
            skip_special_tokens=data.get("skip_special_tokens", True),
            spaces_between_special_tokens=data.get("spaces_between_special_tokens", True),
            add_generation_prompt=data.get("add_generation_prompt", True),
            echo=data.get("echo", False),
            repetition_penalty=data.get("repetition_penalty", 1.0),
            min_p=data.get("min_p", 0.0),
            include_stop_str_in_output=data.get("include_stop_str_in_output", False),
            length_penalty=data.get("length_penalty", 1.0),
        )

        # Generate chat completion using the VLLM engine

        class DummyObject:
            async def is_disconnected(self):
                return False

        async def async_call():
            response = await self.vllm_server.create_chat_completion(request=chat_request, raw_request=DummyObject())
            return response

        chat_completion = asyncio.run(async_call())

        return chat_completion.model_dump() if chat_completion else {"error": "Failed to generate chat completion"}

    def infer_llama_cpp(self, data: dict):
        handler = self.model.create_chat_completion

        response = (
            self.model.create_chat_completion(
                messages=data.get("messages", []),
                functions=data.get("functions"),
                function_call=data.get("function_call"),
                tools=data.get("tools"),
                tool_choice=data.get("tool_choice"),
                temperature=data.get("temperature", 0.2),
                top_p=data.get("top_p", 0.95),
                top_k=data.get("top_k", 40),
                min_p=data.get("min_p", 0.05),
                typical_p=data.get("typical_p", 1.0),
                # stream=data.get("stream", False),
                stop=data.get("stop", []),
                seed=data.get("seed"),
                response_format=data.get("response_format"),
                max_tokens=data.get("max_tokens"),
                presence_penalty=data.get("presence_penalty", 0.0),
                frequency_penalty=data.get("frequency_penalty", 0.0),
                repeat_penalty=data.get("repeat_penalty", 1.1),
                tfs_z=data.get("tfs_z", 1.0),
                mirostat_mode=data.get("mirostat_mode", 0),
                mirostat_tau=data.get("mirostat_tau", 5.0),
                mirostat_eta=data.get("mirostat_eta", 0.1),
                model=data.get("model"),
                logits_processor=data.get("logits_processor"),
                grammar=data.get("grammar"),
                logit_bias=data.get("logit_bias"),
                logprobs=data.get("logprobs"),
                top_logprobs=data.get("top_logprobs"),
            )
            if "messages" in data
            else self.model.create_completion(
                prompt=data.get("prompt"),
                suffix=data.get("suffix", None),
                max_tokens=data.get("max_tokens", 16),
                temperature=data.get("temperature", 0.8),
                top_p=data.get("top_p", 0.95),
                min_p=data.get("min_p", 0.05),
                typical_p=data.get("typical_p", 1.0),
                logprobs=data.get("logprobs", None),
                echo=data.get("echo", False),
                stop=data.get("stop", []),
                frequency_penalty=data.get("frequency_penalty", 0.0),
                presence_penalty=data.get("presence_penalty", 0.0),
                repeat_penalty=data.get("repeat_penalty", 1.1),
                top_k=data.get("top_k", 40),
                # stream=data.get("stream", False),
                seed=data.get("seed", None),
                tfs_z=data.get("tfs_z", 1.0),
                mirostat_mode=data.get("mirostat_mode", 0),
                mirostat_tau=data.get("mirostat_tau", 5.0),
                mirostat_eta=data.get("mirostat_eta", 0.1),
                model=data.get("model", None),
                stopping_criteria=data.get("stopping_criteria", None),
                logits_processor=data.get("logits_processor", None),
                grammar=data.get("grammar", None),
                logit_bias=data.get("logit_bias", None),
            )
        )

        return response


class InstructionInference(TextBulk, _InstructionInference):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
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


class InstructionInferenceStream(TextStream, _InstructionInference):
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
