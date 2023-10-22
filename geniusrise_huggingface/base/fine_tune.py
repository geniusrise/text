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
from abc import abstractmethod
from typing import Dict, Optional, List

import numpy as np
from datasets import Dataset, DatasetDict
from geniusrise import BatchInput, BatchOutput, Bolt, State
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction, Trainer, TrainingArguments
from accelerate import infer_auto_device_map, init_empty_weights
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch
from geniusrise.logging import setup_logger
from geniusrise_huggingface.base.util import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


class HuggingFaceFineTuner(Bolt):
    """
    A bolt for fine-tuning Hugging Face models.

    This bolt uses the Hugging Face Transformers library to fine-tune a pre-trained model.
    It uses the `Trainer` class from the Transformers library to handle the training.
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ) -> None:
        """
        Initialize the bolt.

        Args:
            input (BatchInput): The batch input data.
            output (OutputConfig): The output data.
            state (State): The state manager.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state)
        self.input = input
        self.output = output
        self.state = state

        self.log = setup_logger(self)

    @abstractmethod
    def load_dataset(self, dataset_path: str, **kwargs) -> Dataset | DatasetDict | Optional[Dataset]:
        """
        Load a dataset from a file.

        Args:
            dataset_path (str): The path to the dataset file.
            **kwargs: Additional keyword arguments to pass to the `load_dataset` method.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def preprocess_data(self, **kwargs):
        """Load and preprocess the dataset"""
        try:
            self.input.copy_from_remote()
            train_dataset_path = os.path.join(self.input.get(), "train")
            eval_dataset_path = os.path.join(self.input.get(), "eval")
            self.train_dataset = self.load_dataset(train_dataset_path, **kwargs)
            if self.eval:
                self.eval_dataset = self.load_dataset(eval_dataset_path, **kwargs)
        except Exception as e:
            self.log.exception(f"Failed to preprocess data: {e}")
            raise

    def load_models(
        self,
        model_name: str,
        tokenizer_name: str,
        model_class: str = "AutoModel",
        tokenizer_class: str = "AutoTokenizer",
        device_map: str | dict = "auto",
        precision: str = "bfloat16",
        quantization: Optional[int] = None,
        lora_config: Optional[dict] = None,
        use_accelerate: bool = False,
        accelerate_no_split_module_classes: List[str] = [],
        **kwargs,
    ):
        """Load the model and tokenizer"""
        try:
            # Determine the torch dtype based on precision
            if precision == "float16":
                torch_dtype = torch.float16
            elif precision == "float32":
                torch_dtype = torch.float32
            elif precision == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                raise ValueError("Unsupported precision. Choose from 'float32', 'float16', 'bfloat16'.")

            peft_target_modules = []
            if ":" in model_name:
                model_revision = model_name.split(":")[1]
                model_name = model_name.split(":")[0]
            else:
                model_revision = None
            self.model_name = model_name
            self.log.info(f"Loading model {model_name} and branch {model_revision}")

            with init_empty_weights():
                model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                    model_name, revision=model_revision, device_map=device_map
                )
                known_targets = [
                    v
                    for k, v in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.items()
                    if k.lower() in model_name.lower()
                ]
                if len(known_targets) > 0:
                    peft_target_modules = known_targets[0]
                else:
                    # very generic strategy, may lead to VRAM usage explosion on the wrong model erasing all advantage
                    for name, module in model.named_modules():
                        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)) and "head" not in name:
                            name = name.split(".")[-1]
                            if name not in peft_target_modules:
                                peft_target_modules.append(name)
                self.log.info(f"Targeting these modules for PEFT: {peft_target_modules}")

                if use_accelerate:
                    if precision == "float16":
                        device_map = infer_auto_device_map(
                            model,
                            dtype="float16",
                            no_split_module_classes=accelerate_no_split_module_classes,
                            **kwargs,
                        )
                    elif precision == "bfloat16":
                        device_map = infer_auto_device_map(
                            model,
                            dtype="bfloat16",
                            no_split_module_classes=accelerate_no_split_module_classes,
                            **kwargs,
                        )
                    else:
                        device_map = infer_auto_device_map(
                            model,
                            no_split_module_classes=accelerate_no_split_module_classes,
                            **kwargs,
                        )
                    self.log.info(f"Inferred device map {device_map}")

            if lora_config:
                if len(peft_target_modules) > 0:
                    lora_config = LoraConfig(target_modules=peft_target_modules, **lora_config)
                else:
                    lora_config = LoraConfig(**lora_config)
                self.lora_config = lora_config
            # you cannot fine-tune quantized models without LoRA
            if quantization and not lora_config:
                lora_config = {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                }
                lora_config = LoraConfig(target_modules=peft_target_modules, **lora_config)
            self.log.info(f"LoRA config: {lora_config}")

            # Load model and tokenizer
            if quantization == 8:
                # Use AutoConfig to automatically load the configuration
                if self.model_name.lower() == "local":  # type: ignore
                    self.log.info(f"Loading local model {model_class} : {self.input.get()}")
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        os.path.join(self.input.get(), "/model"),
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        load_in_8bit=True,
                        **kwargs,
                    )
                else:
                    self.log.info(f"Loading from huggingface hub: {model_class} : {model_name}")
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        self.model_name,
                        revision=model_revision,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        load_in_8bit=True,
                        **kwargs,
                    )
            elif quantization == 4:
                # Use AutoConfig to automatically load the configuration
                if self.model_name.lower() == "local":  # type: ignore
                    self.log.info(f"Loading local model {model_class} : {self.input.get()}")
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        os.path.join(self.input.get(), "/model"),
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        load_in_4bit=True,
                        **kwargs,
                    )
                else:
                    self.log.info(f"Loading from huggingface hub: {model_class} : {model_name}")
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        self.model_name,
                        revision=model_revision,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        load_in_4bit=True,
                        **kwargs,
                    )
            else:
                # Use AutoConfig to automatically load the configuration
                if self.model_name.lower() == "local":  # type: ignore
                    self.log.info(f"Loading local model {model_class} : {self.input.get()}")
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        os.path.join(self.input.get(), "/model"),
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        **kwargs,
                    )
                else:
                    self.log.info(f"Loading from huggingface hub: {model_class} : {model_name}")
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        model_name,
                        revision=model_revision,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        **kwargs,
                    )

            if ":" in tokenizer_name:
                tokenizer_revision = tokenizer_name.split(":")[1]
                tokenizer_name = tokenizer_name.split(":")[0]
            else:
                tokenizer_revision = None
            self.tokenizer_name = tokenizer_name

            if tokenizer_name.lower() == "local":  # type: ignore
                self.log.info(f"Loading local tokenizer : {tokenizer_class} : {self.input.get()}")
                self.tokenizer = getattr(__import__("transformers"), str(tokenizer_class)).from_pretrained(
                    os.path.join(self.input.get(), "/model")
                )
            else:
                self.log.info(f"Loading tokenizer from huggingface hub: {tokenizer_class} : {tokenizer_name}")
                self.tokenizer = getattr(__import__("transformers"), str(tokenizer_class)).from_pretrained(
                    tokenizer_name, revision=tokenizer_revision
                )
        except Exception as e:
            self.log.exception(f"Failed to load model: {e}")
            raise

    def upload_to_hf_hub(self):
        """Upload the model and tokenizer to Hugging Face Hub."""
        try:
            if self.model:
                self.model.push_to_hub(
                    repo_id=self.hf_repo_id,
                    commit_message=self.hf_commit_message,
                    token=self.hf_token,
                    private=self.hf_private,
                    create_pr=self.hf_create_pr,
                )
            if self.tokenizer:
                self.tokenizer.push_to_hub(
                    repo_id=self.hf_repo_id,
                    commit_message=self.hf_commit_message,
                    token=self.hf_token,
                    private=self.hf_private,
                    create_pr=self.hf_create_pr,
                )
        except Exception as e:
            self.log.exception(f"Failed to upload model to huggingface hub: {e}")
            raise

    def compute_metrics(self, eval_pred: EvalPrediction) -> Optional[Dict[str, float]] | Dict[str, float]:
        """
        Compute metrics for evaluation. This class implements a simple classification evaluation, tasks should ideally override this.

        Args:
            eval_pred (EvalPrediction): The evaluation predictions.

        Returns:
            dict: The computed metrics.
        """
        predictions, labels = eval_pred
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        labels = labels[0] if isinstance(labels, tuple) else labels
        predictions = np.argmax(predictions, axis=1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_recall_fscore_support(labels, predictions, average="binary")[0],
            "recall": precision_recall_fscore_support(labels, predictions, average="binary")[1],
            "f1": precision_recall_fscore_support(labels, predictions, average="binary")[2],
        }

    def fine_tune(
        self,
        model_name: str,
        tokenizer_name: str,
        num_train_epochs: int,
        per_device_batch_size: int,
        model_class: str = "AutoModel",
        tokenizer_class: str = "AutoTokenizer",
        device_map: str | dict = "auto",
        device: str = "cuda",
        precision: str = "bfloat16",
        quantization: Optional[int] = None,
        lora_config: Optional[dict] = None,
        use_accelerate: bool = False,
        use_trl: bool = False,
        accelerate_no_split_module_classes: List[str] = [],
        eval: bool = False,
        hf_repo_id: Optional[str] = None,
        hf_commit_message: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_private: bool = True,
        hf_create_pr: bool = False,
        **kwargs,
    ):
        """
        Fine-tune the model.

        Args:
            model_name (str): The pre-trained model name.
            tokenizer_name (str): The pre-trained tokenizer name.
            num_train_epochs (int): Total number of training epochs to perform.
            per_device_batch_size (int): Batch size per device during training.
            model_class (str, optional): The model class to use. Defaults to "AutoModel".
            tokenizer_class (str, optional): The tokenizer class to use. Defaults to "AutoTokenizer".
            eval (bool, optional): Whether to evaluate the model after training. Defaults to False.
            hf_repo_id (str, optional): The Hugging Face repo ID. Defaults to None.
            hf_commit_message (str, optional): The Hugging Face commit message. Defaults to None.
            hf_token (str, optional): The Hugging Face token. Defaults to None.
            hf_private (bool, optional): Whether to make the repo private. Defaults to True.
            hf_create_pr (bool, optional): Whether to create a pull request. Defaults to False.
            lora_config (dict, optional): Configuration for PEFT LoRA optimization. Defaults to None.
            use_accelerate (bool, optional): Whether to use accelerate for distributed training. Defaults to False.
            **kwargs: Additional keyword arguments for training.

        Raises:
            Exception: If any step in the fine-tuning process fails.
        """
        try:
            # Save everything
            self.model_name = model_name
            self.tokenizer_name = tokenizer_name
            self.num_train_epochs = num_train_epochs
            self.per_device_batch_size = per_device_batch_size
            self.model_class = model_class
            self.tokenizer_class = tokenizer_class
            self.device_map = device_map
            self.device = device
            self.precision = precision
            self.quantization = quantization
            self.lora_config = lora_config  # type: ignore
            self.use_accelerate = use_accelerate
            self.use_trl = use_trl
            self.accelerate_no_split_module_classes = accelerate_no_split_module_classes
            self.eval = eval
            self.hf_repo_id = hf_repo_id
            self.hf_commit_message = hf_commit_message
            self.hf_token = hf_token
            self.hf_private = hf_private
            self.hf_create_pr = hf_create_pr

            model_kwargs = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}

            self.load_models(
                model_name=self.model_name,
                tokenizer_name=self.tokenizer_name,
                model_class=self.model_class,
                tokenizer_class=self.tokenizer_class,
                device_map=self.device_map,
                precision=self.precision,
                quantization=self.quantization,
                lora_config=self.lora_config,
                use_accelerate=self.use_accelerate,
                accelerate_no_split_module_classes=self.accelerate_no_split_module_classes,
                **model_kwargs,
            )

            if self.tokenizer and not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load dataset
            dataset_kwargs = {k.replace("data_", ""): v for k, v in kwargs.items() if "data_" in k}
            self.preprocess_data(**dataset_kwargs)

            # Separate training and evaluation arguments
            trainer_kwargs = {k.replace("trainer_", ""): v for k, v in kwargs.items() if "trainer_" in k}
            training_kwargs = {k.replace("training_", ""): v for k, v in kwargs.items() if "training_" in k}

            # Create training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.output.output_folder, "model"),
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_batch_size,
                per_device_eval_batch_size=per_device_batch_size,
                **training_kwargs,
            )

            if self.lora_config:
                self.model.enable_input_require_grads()
                self.model = get_peft_model(self.model, peft_config=self.lora_config)

            if self.device and self.model and not self.quantization:
                self.model.to(device)

            # Create trainer
            if use_trl:
                self.model = get_peft_model(self.model, peft_config=self.lora_config)
                trainer = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset if self.eval else None,
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metrics,
                    data_collator=self.data_collator if hasattr(self, "data_collator") else None,
                    peft_config=self.lora_config,
                    **trainer_kwargs,
                )
            else:
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset if self.eval else None,
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metrics,
                    data_collator=self.data_collator if hasattr(self, "data_collator") else None,
                    **trainer_kwargs,
                )

            # Train the model
            trainer.train()
            trainer.save_model()

            if self.eval:
                eval_result = trainer.evaluate()
                self.log.info(f"Evaluation results: {eval_result}")

            # Save the model configuration to Hugging Face Hub if hf_repo_id is not None
            if self.hf_repo_id:
                self.config.save_pretrained(os.path.join(self.output.output_folder, "model"))
                self.upload_to_hf_hub()
        except Exception as e:
            self.log.exception(f"Failed to fine tune model: {e}")
            self.state.set_state(self.id, {"success": False, "exception": str(e)})
            raise
        self.state.set_state(self.id, {"success": True})
