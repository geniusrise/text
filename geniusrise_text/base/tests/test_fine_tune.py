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
import tempfile

import numpy as np
import pytest
import torch
from datasets import load_dataset
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from transformers import DataCollatorForLanguageModeling, EvalPrediction

from geniusrise_text.base import TextFineTuner

# SEQ_CLS = "SEQ_CLS"
# SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
# CAUSAL_LM = "CAUSAL_LM"
# TOKEN_CLS = "TOKEN_CLS"
# QUESTION_ANS = "QUESTION_ANS"
# FEATURE_EXTRACTION = "FEATURE_EXTRACTION"

lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


class TestTextFineTuner(TextFineTuner):
    def load_dataset(self, dataset_path, **kwargs):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")  # Adjust the split as needed
        dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
            ),
            batched=True,
        )
        return dataset

    def data_collator(self, examples):
        return DataCollatorForLanguageModeling(self.tokenizer, mlm=False)(examples)


@pytest.fixture
def bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test", "test-🤗-output")
    state = InMemoryState()

    return TestTextFineTuner(
        input=input,
        output=output,
        state=state,
        evaluate=False,
    )


def test_bolt_init(bolt):
    assert bolt.input is not None
    assert bolt.output is not None
    assert bolt.state is not None


def test_load_dataset(bolt):
    bolt.model_name = "bert-base-uncased"
    bolt.tokenizer_name = "bert-base-uncased"
    bolt.model_class = "AutoModelForCausalLM"
    bolt.tokenizer_class = "BertTokenizer"
    bolt.load_models(
        model_name=bolt.model_name,
        tokenizer_name=bolt.tokenizer_name,
        model_class=bolt.model_class,
        tokenizer_class=bolt.tokenizer_class,
        device_map=None,
    )
    dataset = bolt.load_dataset("fake_path")
    assert dataset is not None
    assert len(dataset) >= 100

    del bolt.model
    del bolt.tokenizer
    torch.cuda.empty_cache()


def test_fine_tune(bolt):
    bolt.fine_tune(
        model_name="bert-base-uncased",
        tokenizer_name="bert-base-uncased",
        num_train_epochs=1,
        per_device_batch_size=2,
        model_class="AutoModelForCausalLM",
        tokenizer_class="BertTokenizer",
        evaluate=False,
        device_map=None,
    )
    bolt.upload_to_hf_hub(
        hf_repo_id="ixaxaar/geniusrise-hf-base-test-repo",
        hf_commit_message="testing base fine tuner",
        hf_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
        hf_private=False,
        hf_create_pr=True,
    )

    # Check that model files are created in the output directory
    assert os.path.isfile(os.path.join(bolt.output.output_folder, "model", "pytorch_model.bin"))
    assert os.path.isfile(os.path.join(bolt.output.output_folder, "model", "config.json"))
    assert os.path.isfile(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))

    del bolt.model
    del bolt.tokenizer
    torch.cuda.empty_cache()


def test_compute_metrics(bolt):
    # Mocking an EvalPrediction object
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)

    metrics = bolt.compute_metrics(eval_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


models = {
    "small": "bigscience/bloom-560m",
    "medium": "meta-llama/Llama-2-7b-hf",
    "large": "mistralai/Mistral-7B-v0.1",
    "4-bit": "TheBloke/Mistral-7B-v0.1-GPTQ:gptq-4bit-32g-actorder_True",
    "8-bit": "TheBloke/OpenHermes-2-Mistral-7B-GPTQ:gptq-8bit-128g-actorder_True",
}


@pytest.mark.parametrize(
    "model, precision, quantization, lora_config, use_accelerate",
    [
        # small
        (models["small"], "float16", None, None, False),
        (models["small"], "float16", None, None, True),
        (models["small"], "float16", None, lora_config, False),
        (models["small"], "float16", None, lora_config, True),
        (models["small"], "float32", None, None, False),
        (models["small"], "float32", None, None, True),
        (models["small"], "float32", None, lora_config, False),
        (models["small"], "float32", None, lora_config, True),
        (models["small"], "bfloat16", None, None, False),
        (models["small"], "bfloat16", None, None, True),
        (models["small"], "bfloat16", None, lora_config, False),
        (models["small"], "bfloat16", None, lora_config, True),
        # small - 4bit
        (models["small"], "float16", 4, lora_config, False),
        (models["small"], "float16", 4, lora_config, True),
        (models["small"], "float32", 4, lora_config, False),
        (models["small"], "float32", 4, lora_config, True),
        (models["small"], "bfloat16", 4, lora_config, False),
        (models["small"], "bfloat16", 4, lora_config, True),
        # small - 8 bit
        (models["small"], "float16", 8, lora_config, False),
        (models["small"], "float16", 8, lora_config, True),
        (models["small"], "float32", 8, lora_config, False),
        (models["small"], "float32", 8, lora_config, True),
        (models["small"], "bfloat16", 8, lora_config, False),
        (models["small"], "bfloat16", 8, lora_config, True),
        # large
        (models["large"], "bfloat16", 4, lora_config, False),
        (models["large"], "bfloat16", 4, lora_config, True),
        (models["large"], "float16", 4, lora_config, False),
        (models["large"], "float16", 4, lora_config, True),
        (models["large"], "float32", 4, lora_config, False),
        (models["large"], "float32", 4, lora_config, True),
        # 4 bit
        (models["4-bit"], "float16", None, lora_config, False),
        # 8 bit
        (models["8-bit"], "float16", None, lora_config, False),
        (models["8-bit"], "float16", None, lora_config, True),
    ],
)
def test_fine_tune_options(bolt, model, precision, quantization, lora_config, use_accelerate):
    use_trl = False

    if use_trl:
        bolt.fine_tune(
            model_name=model,
            tokenizer_name=model,
            model_class="AutoModelForCausalLM",
            tokenizer_class="AutoTokenizer",
            num_train_epochs=1,
            per_device_batch_size=2,
            precision=precision,
            quantization=quantization,
            lora_config=lora_config,
            use_accelerate=use_accelerate,
            device_map="auto" if "GPTQ" in model else None,
            trainer_packing=False if lora_config is not None else None,
            trainer_dataset_text_field="text" if lora_config is not None else None,
        )
    else:
        bolt.fine_tune(
            model_name=model,
            tokenizer_name=model,
            model_class="AutoModelForCausalLM",
            tokenizer_class="AutoTokenizer",
            num_train_epochs=1,
            per_device_batch_size=2,
            precision=precision,
            quantization=quantization,
            lora_config=lora_config,
            use_accelerate=use_accelerate,
            device_map="auto" if "GPTQ" in model else None,
        )

    # Verify the model has been fine-tuned by checking the existence of model files
    assert os.path.exists(os.path.join(bolt.output.output_folder, "model", "pytorch_model.bin")) or os.path.exists(
        os.path.join(bolt.output.output_folder, "model", "adapter_model.bin")
    )
    assert os.path.exists(os.path.join(bolt.output.output_folder, "model", "config.json")) or os.path.exists(
        os.path.join(bolt.output.output_folder, "model", "adapter_config.json")
    )
    assert os.path.exists(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))

    # Clear the output directory for the next test
    try:
        os.remove(os.path.join(bolt.output.output_folder, "model", "pytorch_model.bin"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "adapter_model.bin"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "config.json"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "adapter_config.json"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))
    except Exception as _:
        pass

    del bolt.model
    del bolt.tokenizer
    torch.cuda.empty_cache()
