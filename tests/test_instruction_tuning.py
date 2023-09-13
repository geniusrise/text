# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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

from geniusrise_huggingface import HuggingFaceInstructionTuningFineTuner


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


@pytest.fixture
def instruction_tuning_bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    input = BatchInput(input_dir, "geniusrise-test-bucket", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "test-🤗-output")
    state = InMemoryState()
    klass = HuggingFaceInstructionTuningFineTuner(
        input=input,
        output=output,
        state=state,
    )
    klass.model_class = "BartForConditionalGeneration"
    klass.model_name = "facebook/bart-base"
    klass.tokenizer_class = "BartTokenizer"
    klass.tokenizer_name = "facebook/bart-base"
    return klass


def test_instruction_tuning_bolt_init(instruction_tuning_bolt):
    instruction_tuning_bolt.load_models()

    assert instruction_tuning_bolt.model is not None
    assert instruction_tuning_bolt.tokenizer is not None
    assert instruction_tuning_bolt.input is not None
    assert instruction_tuning_bolt.output is not None
    assert instruction_tuning_bolt.state is not None


def test_load_dataset_all_formats(instruction_tuning_bolt, dataset_file):
    tmpdir, ext = dataset_file
    dataset_path = os.path.join(tmpdir, "train")

    instruction_tuning_bolt.load_models()
    dataset = instruction_tuning_bolt.load_dataset(dataset_path)
    assert dataset is not None
    assert len(dataset) == 10


# Test for fine-tuning
def test_instruction_tuning_bolt_fine_tune(instruction_tuning_bolt, dataset_file):
    tmpdir, ext = dataset_file
    instruction_tuning_bolt.input.input_folder = tmpdir

    instruction_tuning_bolt.fine_tune(
        model_name="facebook/bart-base",
        tokenizer_name="facebook/bart-base",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        model_class="BartForConditionalGeneration",
        tokenizer_class="BartTokenizer",
        eval=True,
    )

    output_dir = instruction_tuning_bolt.output.output_folder
    assert os.path.isfile(os.path.join(output_dir + "/model", "pytorch_model.bin"))
    assert os.path.isfile(os.path.join(output_dir + "/model", "config.json"))
    assert os.path.isfile(os.path.join(output_dir + "/model", "training_args.bin"))


# Test for computing metrics
def test_instruction_tuning_bolt_compute_metrics(instruction_tuning_bolt):
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    instruction_tuning_bolt.load_models()
    metrics = instruction_tuning_bolt.compute_metrics(eval_pred)
    assert "bleu" in metrics
