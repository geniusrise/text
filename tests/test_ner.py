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
import yaml  # type: ignore
from datasets import Dataset
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from pyarrow import feather
from pyarrow import parquet as pq
from transformers import EvalPrediction

from geniusrise_huggingface.ner import HuggingFaceNamedEntityRecognitionFineTuner


# Helper function to create synthetic data in different formats
def create_dataset_in_format(directory, ext):
    os.makedirs(directory, exist_ok=True)
    data = [{"tokens": ["This", "is", "a", "test"], "ner_tags": [0, 1, 0, 1]} for _ in range(10)]
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
            ET.SubElement(record, "tokens").text = " ".join(item["tokens"])
            ET.SubElement(record, "ner_tags").text = " ".join(map(str, item["ner_tags"]))
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
        "feather",
    ]
)
def dataset_file(request, tmpdir):
    ext = request.param
    create_dataset_in_format(tmpdir + "/train", ext)
    create_dataset_in_format(tmpdir + "/eval", ext)
    return tmpdir, ext


@pytest.fixture
def ner_bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test-bucket", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "test-🤗-output")
    state = InMemoryState()

    klass = HuggingFaceNamedEntityRecognitionFineTuner(
        input=input,
        output=output,
        state=state,
    )
    klass.model_class = "BertForTokenClassification"
    klass.model_name = "bert-base-uncased"
    klass.tokenizer_class = "BertTokenizerFast"
    klass.tokenizer_name = "bert-base-uncased"

    return klass


def test_ner_bolt_init(ner_bolt):
    ner_bolt.load_models()

    assert ner_bolt.model is not None
    assert ner_bolt.tokenizer is not None
    assert ner_bolt.input is not None
    assert ner_bolt.output is not None
    assert ner_bolt.state is not None


def test_load_dataset_all_formats(ner_bolt, dataset_file):
    tmpdir, ext = dataset_file
    dataset_path = os.path.join(tmpdir, "train")

    ner_bolt.load_models()
    dataset = ner_bolt.load_dataset(dataset_path, label_list=[0, 1])
    assert dataset is not None
    assert len(dataset) == 10


# Test for fine-tuning
def test_ner_bolt_fine_tune(ner_bolt, dataset_file):
    tmpdir, ext = dataset_file
    ner_bolt.input.input_folder = tmpdir

    ner_bolt.fine_tune(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        model_class="BertForTokenClassification",
        model_name="bert-base-uncased",
        tokenizer_class="BertTokenizerFast",
        tokenizer_name="bert-base-uncased",
        data_label_list=[0, 1],
    )

    output_dir = ner_bolt.output.output_folder
    assert os.path.isfile(os.path.join(output_dir + "/model", "pytorch_model.bin"))
    assert os.path.isfile(os.path.join(output_dir + "/model", "config.json"))
    assert os.path.isfile(os.path.join(output_dir + "/model", "training_args.bin"))


# Test for computing metrics
def test_ner_bolt_compute_metrics(ner_bolt):
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    metrics = ner_bolt.compute_metrics(eval_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
