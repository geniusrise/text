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

import json
import os
import sqlite3
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
import torch
import yaml  # type: ignore
from datasets import Dataset
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from pyarrow import feather
from pyarrow import parquet as pq
from transformers import EvalPrediction

from geniusrise_text.instruction.fine_tune import InstructionFineTuner

lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


# Helper function to create synthetic data in different formats
def create_dataset_in_format(directory, ext):
    os.makedirs(directory, exist_ok=True)
    data = [{"instruction": f"instruction_{i}", "output": f"output_{i}"} for i in range(10)]
    df = pd.DataFrame(data)

    if ext == "huggingface":
        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk(directory)
    elif ext == "csv":
        df.to_csv(os.path.join(directory, "data.csv"), index=False)
    elif ext == "jsonl":
        with open(os.path.join(directory, "data.jsonl"), "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    elif ext == "parquet":
        pq.write_table(feather.Table.from_pandas(df), os.path.join(directory, "data.parquet"))
    elif ext == "json":
        with open(os.path.join(directory, "data.json"), "w") as f:
            json.dump(data, f)
    elif ext == "xml":
        root = ET.Element("root")
        for item in data:
            record = ET.SubElement(root, "record")
            ET.SubElement(record, "instruction").text = item["instruction"]
            ET.SubElement(record, "output").text = item["output"]
        tree = ET.ElementTree(root)
        tree.write(os.path.join(directory, "data.xml"))
    elif ext == "yaml":
        with open(os.path.join(directory, "data.yaml"), "w") as f:
            yaml.dump(data, f)
    elif ext == "tsv":
        df.to_csv(os.path.join(directory, "data.tsv"), index=False, sep="\t")
    elif ext == "xlsx":
        df.to_excel(os.path.join(directory, "data.xlsx"), index=False)
    elif ext == "db":
        conn = sqlite3.connect(os.path.join(directory, "data.db"))
        df.to_sql("dataset_table", conn, if_exists="replace", index=False)
        conn.close()
    elif ext == "feather":
        feather.write_feather(df, os.path.join(directory, "data.feather"))


# Fixtures for each file type
@pytest.fixture(
    params=[
        "huggingface",
        "csv",
        "jsonl",
        "parquet",
        "json",
        "xml",
        "yaml",
        "tsv",
        "xlsx",
        "db",
        "feather",
    ]
)
def dataset_file(request, tmpdir):
    ext = request.param
    create_dataset_in_format(tmpdir + "/train", ext)
    create_dataset_in_format(tmpdir + "/eval", ext)
    return tmpdir, ext


MODELS_TO_TEST = {
    # fmt: off
    "small": "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
    # fmt: on
}


# Fixture for models
@pytest.fixture(params=MODELS_TO_TEST.items())
def model(request):
    return request.param


@pytest.fixture
def instruction_tuning_bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    input = BatchInput(input_dir, "geniusrise-test", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test", "test-🤗-output")
    state = InMemoryState()
    klass = InstructionFineTuner(
        input=input,
        output=output,
        state=state,
    )
    return klass


def test_instruction_tuning_bolt_init(instruction_tuning_bolt, model):
    name, model_name = model
    tokenizer_name = model_name
    model_class = "AutoModelForCausalLM"
    tokenizer_class = "AutoTokenizer"

    instruction_tuning_bolt.load_models(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
    )

    assert instruction_tuning_bolt.model is not None
    assert instruction_tuning_bolt.tokenizer is not None
    assert instruction_tuning_bolt.input is not None
    assert instruction_tuning_bolt.output is not None
    assert instruction_tuning_bolt.state is not None


def test_load_dataset_all_formats(instruction_tuning_bolt, dataset_file, model):
    name, model_name = model
    tokenizer_name = model_name
    model_class = "AutoModelForCausalLM"
    tokenizer_class = "AutoTokenizer"

    instruction_tuning_bolt.load_models(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
    )

    tmpdir, ext = dataset_file
    dataset_path = os.path.join(tmpdir, "train")
    dataset = instruction_tuning_bolt.load_dataset(dataset_path)
    assert dataset is not None
    assert len(dataset) == 10


# Models to test
models = {
    # fmt: off
    "small": "PY007/TinyLlama-1.1B-Chat-v0.3",
    "medium": "HuggingFaceH4/zephyr-7b-beta",
    "large": "mistralai/Mistral-7B-Instruct-v0.1",
    # mistral
    "4-bit-mistral": "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ:gptq-4bit-32g-actorder_True",
    "8-bit-mistral": "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ:gptq-8bit-32g-actorder_True",
    "4-bit-mistral-code": "TheBloke/Mistral-7B-Code-16K-qlora-GPTQ:gptq-4bit-32g-actorder_True",
    "4-bit-mistral-32k": "TheBloke/Mistral-7B-Phibrarian-32K-GPTQ",
    # zephyr
    "4-bit-zephyr": "TheBloke/zephyr-7B-beta-GPTQ:gptq-4bit-32g-actorder_True",
    "8-bit-zephyr": "TheBloke/zephyr-7B-beta-GPTQ:gptq-8bit-32g-actorder_True",
    # falcon
    "4-bit-falcon": "TheBloke/falcon-7b-instruct-GPTQ",
    # llama-based
    "4-bit-llama-2": "TheBloke/Llama-2-7b-Chat-GPTQ:gptq-4bit-32g-actorder_True",
    "4-bit-llama-2-32k": "TheBloke/Llama-2-7B-32K-Instruct-GPTQ:gptq-4bit-32g-actorder_True",
    "4-bit-wizard": "TheBloke/WizardLM-7B-uncensored-GPTQ",
    "4-bit-vicuna": "TheBloke/vicuna-7B-v1.5-GPTQ:gptq-4bit-32g-actorder_True",
    "4-bit-wizard-vicuna": "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ",
    "4-bit-yarn-64k": "TheBloke/Yarn-Llama-2-7B-64K-GPTQ:gptq-4bit-32g-actorder_True",
    "4-bit-yarn-128k": "TheBloke/Yarn-Llama-2-7B-128K-GPTQ:gptq-4bit-32g-actorder_True",
    # fmt: on
}


# Test for fine-tuning
@pytest.mark.parametrize(
    "model_name, precision, quantization, lora_config, use_accelerate",
    [
        # small
        (models["small"], "float16", None, None, False),
        (models["small"], "float16", None, None, True),
        (models["small"], "float16", None, lora_config, False),
        (models["small"], "float16", None, lora_config, True),
        (models["small"], "bfloat16", None, None, False),
        (models["small"], "bfloat16", None, None, True),
        (models["small"], "bfloat16", None, lora_config, False),
        (models["small"], "bfloat16", None, lora_config, True),
        # small - 4bit
        (models["small"], "float16", 4, lora_config, False),
        (models["small"], "float16", 4, lora_config, True),
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
        # # 4 bit
        (models["4-bit-mistral"], "float16", None, lora_config, False),
        (models["4-bit-mistral-code"], "float16", None, lora_config, False),
        (models["4-bit-mistral-32k"], "float16", None, lora_config, False),
        (models["4-bit-zephyr"], "float16", None, lora_config, False),
        (models["4-bit-falcon"], "float16", None, lora_config, False),
        (models["4-bit-llama-2"], "float16", None, lora_config, False),
        (models["4-bit-llama-2-32k"], "float16", None, lora_config, False),
        (models["4-bit-wizard"], "float16", None, lora_config, False),
        (models["4-bit-vicuna"], "float16", None, lora_config, False),
        (models["4-bit-wizard-vicuna"], "float16", None, lora_config, False),
        (models["4-bit-yarn-64k"], "float16", None, lora_config, False),
        (models["4-bit-yarn-128k"], "float16", None, lora_config, False),
        # # 8 bit
        (models["8-bit-mistral"], "float16", None, lora_config, False),
        (models["8-bit-zephyr"], "float16", None, lora_config, False),
    ],
)
def test_instruction_tuning_bolt_fine_tune(
    instruction_tuning_bolt, dataset_file, model_name, precision, quantization, lora_config, use_accelerate
):
    try:
        tokenizer_name = model_name

        tmpdir, ext = dataset_file
        instruction_tuning_bolt.input.input_folder = tmpdir

        instruction_tuning_bolt.fine_tune(
            model_name=model_name,
            tokenizer_name=model_name,
            model_class="AutoModelForCausalLM",
            tokenizer_class="AutoTokenizer",
            num_train_epochs=1,
            per_device_batch_size=2,
            precision=precision,
            quantization=quantization,
            lora_config=lora_config,
            use_accelerate=use_accelerate,
            device_map="auto" if "GPTQ" in model_name else None,
            data_masked=False,
        )
        output_dir = instruction_tuning_bolt.output.output_folder
        assert os.path.exists(
            os.path.join(instruction_tuning_bolt.output.output_folder, "model", "pytorch_model.bin")
        ) or os.path.exists(os.path.join(instruction_tuning_bolt.output.output_folder, "model", "adapter_model.bin"))
        assert os.path.exists(
            os.path.join(instruction_tuning_bolt.output.output_folder, "model", "config.json")
        ) or os.path.exists(os.path.join(instruction_tuning_bolt.output.output_folder, "model", "adapter_config.json"))
        assert os.path.exists(os.path.join(instruction_tuning_bolt.output.output_folder, "model", "training_args.bin"))

        del instruction_tuning_bolt.model
        del instruction_tuning_bolt.tokenizer
        torch.cuda.empty_cache()

        try:
            os.remove(os.path.join(instruction_tuning_bolt.output.output_folder, "model", "pytorch_model.bin"))
            os.remove(os.path.join(instruction_tuning_bolt.output.output_folder, "model", "adapter_model.bin"))
            os.remove(os.path.join(instruction_tuning_bolt.output.output_folder, "model", "config.json"))
            os.remove(os.path.join(instruction_tuning_bolt.output.output_folder, "model", "adapter_config.json"))
            os.remove(os.path.join(instruction_tuning_bolt.output.output_folder, "model", "training_args.bin"))
        except Exception as _:
            pass

    except Exception as e:
        del instruction_tuning_bolt.model
        del instruction_tuning_bolt.tokenizer
        torch.cuda.empty_cache()
        raise


# Test for computing metrics
def test_instruction_tuning_bolt_compute_metrics(instruction_tuning_bolt, model):
    name, model_name = model
    tokenizer_name = model_name
    model_class = "AutoModelForCausalLM"
    tokenizer_class = "AutoTokenizer"

    instruction_tuning_bolt.load_models(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        device_map="cuda:0",
    )

    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    instruction_tuning_bolt.load_models()
    metrics = instruction_tuning_bolt.compute_metrics(eval_pred)
    assert "bleu" in metrics
