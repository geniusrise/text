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

from geniusrise_text.qa.fine_tune import QAFineTuner

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
    data = [
        {
            "context": f"context_{i}",
            "question": f"question_{i}",
            "answers": [{"answer_start": [0], "text": [f"answer_{i}"]}],
        }
        for i in range(10)
    ]
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
            ET.SubElement(record, "context").text = item["context"]
            ET.SubElement(record, "question").text = item["question"]
            ET.SubElement(record, "answers").text = str(item["answers"])
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
        df["answers"] = df["answers"].apply(str)
        df.to_sql("dataset_table", conn, if_exists="replace", index=False)
        conn.close()
    elif ext == "feather":
        feather.write_feather(df, os.path.join(directory, "data.feather"))


# Fixtures for each file type
@pytest.fixture(
    params=[
        "huggingface",
        "csv",
        # "db",
        # "xml",
        # "jsonl",
        # "parquet",
        # "json",
        # "yaml",
        # "tsv",
        # "xlsx",
        # "feather",
    ]
)
def dataset_file(request, tmpdir):
    ext = request.param
    create_dataset_in_format(tmpdir + "/train", ext)
    create_dataset_in_format(tmpdir + "/eval", ext)
    return tmpdir, ext


MODELS_TO_TEST = {
    # fmt: off
    "small": "facebook/bart-base",
    # fmt: on
}


# Fixture for models
@pytest.fixture(params=MODELS_TO_TEST.items())
def model(request):
    return request.param


@pytest.fixture
def question_answering_bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    input = BatchInput(input_dir, "geniusrise-test", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test", "test-🤗-output")
    state = InMemoryState()
    klass = QAFineTuner(
        input=input,
        output=output,
        state=state,
    )
    return klass


# def test_question_answering_bolt_init(question_answering_bolt, model):
#     name, model_name = model
#     tokenizer_name = model_name
#     model_class = "AutoModelForQuestionAnswering"
#     tokenizer_class = "AutoTokenizer"

#     question_answering_bolt.load_models(
#         model_name=model_name,
#         tokenizer_name=tokenizer_name,
#         model_class=model_class,
#         tokenizer_class=tokenizer_class,
#         device_map=None,
#     )

#     assert question_answering_bolt.model is not None
#     assert question_answering_bolt.tokenizer is not None
#     assert question_answering_bolt.input is not None
#     assert question_answering_bolt.output is not None
#     assert question_answering_bolt.state is not None


# def test_load_dataset_all_formats(question_answering_bolt, dataset_file, model):
#     tmpdir, ext = dataset_file
#     dataset_path = os.path.join(tmpdir, "train")

#     name, model_name = model
#     tokenizer_name = model_name
#     model_class = "AutoModelForQuestionAnswering"
#     tokenizer_class = "AutoTokenizer"

#     question_answering_bolt.load_models(
#         model_name=model_name,
#         tokenizer_name=tokenizer_name,
#         model_class=model_class,
#         tokenizer_class=tokenizer_class,
#         device_map=None,
#     )

#     dataset = question_answering_bolt.load_dataset(dataset_path)
#     assert dataset is not None
#     assert len(dataset) == 10


# Models to test
models = {
    # fmt: off
    "bart": "facebook/bart-large-cnn",
    "distillbart": "sshleifer/distilbart-cnn-12-6",
    "flan": "google/flan-t5-xl",
    "t5": "google/t5-v1_1-xl",
    "mt5": "google/umt5-xl",
    "pegasus": "google/pegasus-x-large",
    "pegasus-news": "google/pegasus-multi_news",
    "bigbird-arxiv": "google/bigbird-pegasus-large-arxiv",
    "bigbird-bigpatent": "google/bigbird-pegasus-large-bigpatent",
    "bigbird-pubmed": "google/bigbird-pegasus-large-pubmed",
    # fmt: on
}


# Test for fine-tuning
@pytest.mark.parametrize(
    "model_name, precision, quantization, lora_config, use_accelerate",
    [
        # small
        (models["bart"], "bfloat16", None, None, False),
        (models["distillbart"], "bfloat16", None, None, False),
        (models["flan"], "bfloat16", 8, lora_config, False),
        (models["t5"], "bfloat16", 8, lora_config, False),
        (models["mt5"], "bfloat16", 8, lora_config, False),
        (models["pegasus"], "bfloat16", 8, lora_config, False),
        (models["pegasus-news"], "bfloat16", None, None, False),
        (models["bigbird-arxiv"], "bfloat16", None, None, False),
        (models["bigbird-bigpatent"], "bfloat16", None, None, False),
        (models["bigbird-pubmed"], "bfloat16", None, None, False),
    ],
)
def test_question_answering_bolt_fine_tune(
    question_answering_bolt, dataset_file, model_name, precision, quantization, lora_config, use_accelerate
):
    try:
        tokenizer_name = model_name

        tmpdir, ext = dataset_file
        question_answering_bolt.input.input_folder = tmpdir

        question_answering_bolt.fine_tune(
            model_name=model_name,
            tokenizer_name=model_name,
            model_class="AutoModelForQuestionAnswering",
            tokenizer_class="AutoTokenizer",
            num_train_epochs=1,
            per_device_batch_size=2,
            precision=precision,
            quantization=quantization,
            lora_config=lora_config,
            use_accelerate=use_accelerate,
            device_map=None,
        )
        output_dir = question_answering_bolt.output.output_folder
        assert os.path.exists(
            os.path.join(question_answering_bolt.output.output_folder, "model", "pytorch_model.bin")
        ) or os.path.exists(os.path.join(question_answering_bolt.output.output_folder, "model", "adapter_model.bin"))
        assert os.path.exists(
            os.path.join(question_answering_bolt.output.output_folder, "model", "config.json")
        ) or os.path.exists(os.path.join(question_answering_bolt.output.output_folder, "model", "adapter_config.json"))
        assert os.path.exists(os.path.join(question_answering_bolt.output.output_folder, "model", "training_args.bin"))

        del question_answering_bolt.model
        del question_answering_bolt.tokenizer
        torch.cuda.empty_cache()

        try:
            os.remove(os.path.join(question_answering_bolt.output.output_folder, "model", "pytorch_model.bin"))
            os.remove(os.path.join(question_answering_bolt.output.output_folder, "model", "adapter_model.bin"))
            os.remove(os.path.join(question_answering_bolt.output.output_folder, "model", "config.json"))
            os.remove(os.path.join(question_answering_bolt.output.output_folder, "model", "adapter_config.json"))
            os.remove(os.path.join(question_answering_bolt.output.output_folder, "model", "training_args.bin"))
        except Exception as _:
            pass

    except Exception as e:
        del question_answering_bolt.model
        del question_answering_bolt.tokenizer
        torch.cuda.empty_cache()
        raise


# Test for computing metrics
def test_question_answering_bolt_compute_metrics(question_answering_bolt):
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    metrics = question_answering_bolt.compute_metrics(eval_pred)
    assert "accuracy" in metrics
