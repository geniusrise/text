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

import os
import pytest

import torch

# import transformers
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from geniusrise_huggingface.base.api import HuggingFaceAPI


# Strategies
strategies = [
    "generate",
    "greedy_search",
    "contrastive_search",
    "sample",
    "beam_search",
    "beam_sample",
    "group_beam_search",
    "constrained_beam_search",
]

# Parameters controlling the length of the output
max_length = [20, 30, 40]
min_length = [0, 10, 20]
early_stopping = [False, True, "never"]
max_time = [None, 2.0, 5.0]

# Parameters controlling the generation strategy used
do_sample = [False, True]
num_beams = [1, 2, 4]
num_beam_groups = [1, 2]

# Parameters for manipulation of the model output logits
temperature = [1.0, 0.7, 0.5]
top_k = [50, 20, 10]
top_p = [1.0, 0.9, 0.8]
repetition_penalty = [1.0, 1.5, 2.0]
length_penalty = [1.0, 0.5, 1.5]
no_repeat_ngram_size = [0, 2, 3]


@pytest.fixture(
    params=[
        # model_name, model_class_name, tokenizer_class_name, use_cuda, precision, quantization, device_map, max_memory,torchscript
        # fmt: off
        ("gpt2", "AutoModelForCausalLM", "AutoTokenizer", True, "float16", 0, None, None, False),
        ("gpt2", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", 0, None, None, False),
        ("gpt2", "AutoModelForCausalLM", "AutoTokenizer", False, "float32", 0, None, None, False),
        ("bigscience/bloom-560m", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", 0, None, None, False),
        ("meta-llama/Llama-2-7b-hf", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", 0, None, None, False),
        ("mistralai/Mistral-7B-v0.1", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", 0, "cuda:0", None, False),
        ("mistralai/Mistral-7B-v0.1", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", 4, "cuda:0", None, False),
        ("mistralai/Mistral-7B-v0.1", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", 8, "cuda:0", None, False),
        ("mistralai/Mistral-7B-v0.1", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", 0, "cuda:0", None, True),
        ("mistralai/Mistral-7B-v0.1", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", 4, "cuda:0", None, True),
        ("mistralai/Mistral-7B-v0.1", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", 8, "cuda:0", None, True),
        ("TheBloke/Mistral-7B-v0.1-GPTQ:gptq-4bit-32g-actorder_True", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", None, "cuda:0", None, False),
        ("TheBloke/OpenHermes-2-Mistral-7B-GPTQ:gptq-8bit-128g-actorder_True", "AutoModelForCausalLM", "AutoTokenizer", True, "bfloat16", None, "cuda:0", None, False),
        # fmt: on
    ]
)
def model_config(request):
    return request.param


# Fixtures to initialize HuggingFaceAPI instance
@pytest.fixture
def hfa():
    input_dir = "./input_dir"
    output_dir = "./output_dir"

    input = BatchInput(input_dir, "geniusrise-test", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test", "api_output")
    state = InMemoryState()

    hfa = HuggingFaceAPI(
        input=input,
        output=output,
        state=state,
    )
    yield hfa  # provide the fixture value

    # cleanup
    if os.path.exists(input_dir):
        os.rmdir(input_dir)
    if os.path.exists(output_dir):
        os.rmdir(output_dir)


def test_load_models(hfa, model_config):
    (
        model_name,
        model_class_name,
        tokenizer_class_name,
        use_cuda,
        precision,
        quantization,
        device_map,
        max_memory,
        torchscript,
    ) = model_config

    if ":" in model_name:
        _model_name = model_name
        model_revision = _model_name.split(":")[1]
        model_name = _model_name.split(":")[0]
        tokenizer_revision = _model_name.split(":")[1]
        tokenizer_name = _model_name.split(":")[0]
    else:
        model_revision = None
        tokenizer_revision = None

    model, tokenizer = hfa.load_models(
        model_name=model_name,
        model_revision=model_revision,
        tokenizer_name=model_name,
        tokenizer_revision=tokenizer_revision,
        model_class_name=model_class_name,
        tokenizer_class_name=tokenizer_class_name,
        use_cuda=use_cuda,
        precision=precision,
        quantization=quantization,
        device_map=device_map,
        max_memory=max_memory,
        torchscript=torchscript,
    )
    assert model is not None
    assert tokenizer is not None
    assert len(list(model.named_modules())) > 0

    del model
    del tokenizer
    torch.cuda.empty_cache()


# def test_generate(hfa, model_config):
#     (
#         model_name,
#         model_class_name,
#         tokenizer_class_name,
#         use_cuda,
#         precision,
#         device_map,
#         max_memory,
#         torchscript,
#     ) = model_config
#     hfa.load_models(
#         model_name=model_name,
#         model_class_name=model_class_name,
#         tokenizer_class_name=tokenizer_class_name,
#         use_cuda=use_cuda,
#         precision=precision,
#         device_map=device_map,
#         max_memory=max_memory,
#         torchscript=torchscript,
#     )
#     generated_text = hfa.generate("Once upon a time")
#     assert generated_text is not None
#     assert isinstance(generated_text, str)


# @pytest.mark.parametrize(
#     "strategy,max_length,min_length,early_stopping,max_time,"
#     "do_sample,num_beams,num_beam_groups,temperature,top_k,"
#     "top_p,repetition_penalty,length_penalty,no_repeat_ngram_size",
#     [
#         (s, ml, mil, es, mt, ds, nb, nbg, t, tk, tp, rp, lp, nrns)
#         for s in strategies
#         for ml in max_length
#         for mil in min_length
#         for es in early_stopping
#         for mt in max_time
#         for ds in do_sample
#         for nb in num_beams
#         for nbg in num_beam_groups
#         for t in temperature
#         for tk in top_k
#         for tp in top_p
#         for rp in repetition_penalty
#         for lp in length_penalty
#         for nrns in no_repeat_ngram_size
#     ],
# )
# def test_generate_strategies(
#     hfa,
#     strategy,
#     max_length,
#     min_length,
#     early_stopping,
#     max_time,
#     do_sample,
#     num_beams,
#     num_beam_groups,
#     temperature,
#     top_k,
#     top_p,
#     repetition_penalty,
#     length_penalty,
#     no_repeat_ngram_size,
#     model_config,
# ):
#     (
#         model_name,
#         model_class_name,
#         tokenizer_class_name,
#         use_cuda,
#         precision,
#         device_map,
#         max_memory,
#         torchscript,
#     ) = model_config
#     hfa.load_models(
#         model_name=model_name,
#         model_class_name=model_class_name,
#         tokenizer_class_name=tokenizer_class_name,
#         use_cuda=use_cuda,
#         precision=precision,
#         device_map=device_map,
#         max_memory=max_memory,
#         torchscript=torchscript,
#     )

#     generated_text = hfa.generate(
#         input_text="Once upon a time",
#         strategy=strategy,
#         max_length=max_length,
#         min_length=min_length,
#         early_stopping=early_stopping,
#         max_time=max_time,
#         do_sample=do_sample,
#         num_beams=num_beams,
#         num_beam_groups=num_beam_groups,
#         temperature=temperature,
#         top_k=top_k,
#         top_p=top_p,
#         repetition_penalty=repetition_penalty,
#         length_penalty=length_penalty,
#         no_repeat_ngram_size=no_repeat_ngram_size,
#     )

#     assert generated_text is not None
#     assert isinstance(generated_text, str)
