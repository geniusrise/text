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
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from datasets import Dataset, DatasetDict
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoConfig, EvalPrediction, Trainer, TrainingArguments
from trl import SFTTrainer

from geniusrise_text.base.communication import send_fine_tuning_email
from geniusrise_text.base.util import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


class TextFineTuner(Bolt):
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
            output (BatchOutput): The output data.
            state (State): The state manager.
            evaluate (bool, optional): Whether to evaluate the model. Defaults to False.
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
            split (str, optional): The split to load. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the `load_dataset` method.

        Returns:
            Union[Dataset, DatasetDict, None]: The loaded dataset.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def preprocess_data(self, **kwargs):
        """Load and preprocess the dataset"""
        try:
            if self.use_huggingface_dataset:
                _dataset = self.load_dataset(self.huggingface_dataset, **kwargs)
                if self.evaluate:
                    _dataset = _dataset.train_test_split(test_size=0.2)
                    self.train_dataset = _dataset["train"]
                    self.eval_dataset = _dataset["test"]
                else:
                    self.train_dataset = _dataset["train"]
                    self.eval_dataset = None
            elif self.evaluate:
                train_dataset_path = os.path.join(self.input.get(), "train")
                eval_dataset_path = os.path.join(self.input.get(), "test")
                self.train_dataset = self.load_dataset(train_dataset_path, **kwargs)
                self.eval_dataset = self.load_dataset(eval_dataset_path, **kwargs)
            else:
                self.train_dataset = self.load_dataset(self.input.get(), **kwargs)
                self.eval_dataset = None
        except Exception as e:
            self.log.exception(f"Failed to preprocess data: {e}")
            raise e

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
        """
        Load the model and tokenizer.

        Args:
            model_name (str): The name of the model to be loaded.
            tokenizer_name (str, optional): The name of the tokenizer to be loaded. Defaults to None.
            model_class (str, optional): The class of the model. Defaults to "AutoModel".
            tokenizer_class (str, optional): The class of the tokenizer. Defaults to "AutoTokenizer".
            device (Union[str, torch.device], optional): The device to be used. Defaults to "cuda".
            precision (str, optional): The precision to be used. Choose from 'float32', 'float16', 'bfloat16'. Defaults to "float32".
            quantization (Optional[int], optional): The quantization to be used. Defaults to None.
            lora_config (Optional[dict], optional): The LoRA configuration to be used. Defaults to None.
            use_accelerate (bool, optional): Whether to use accelerate. Defaults to False.
            accelerate_no_split_module_classes (List[str], optional): The list of no split module classes to be used. Defaults to [].
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If an unsupported precision is chosen.

        Returns:
            None
        """
        try:
            # Determine the torch dtype based on precision
            if precision == "float32":
                torch_dtype = torch.float32
            elif precision == "float":
                torch_dtype = torch.float
            elif precision == "float64":
                torch_dtype = torch.float64
            elif precision == "double":
                torch_dtype = torch.double
            elif precision == "float16":
                torch_dtype = torch.float16
            elif precision == "bfloat16":
                torch_dtype = torch.bfloat16
            elif precision == "half":
                torch_dtype = torch.half
            elif precision == "uint8":
                torch_dtype = torch.uint8
            elif precision == "int8":
                torch_dtype = torch.int8
            elif precision == "int16":
                torch_dtype = torch.int16
            elif precision == "short":
                torch_dtype = torch.short
            elif precision == "int32":
                torch_dtype = torch.int32
            elif precision == "int":
                torch_dtype = torch.int
            elif precision == "int64":
                torch_dtype = torch.int64
            elif precision == "quint8":
                torch_dtype = torch.quint8
            elif precision == "qint8":
                torch_dtype = torch.qint8
            elif precision == "qint32":
                torch_dtype = torch.qint32
            else:
                torch_dtype = None

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

            # Create the LoRA config for PEFT
            if lora_config:
                if len(peft_target_modules) > 0:
                    lora_config = LoraConfig(target_modules=peft_target_modules, **lora_config)
                else:
                    lora_config = LoraConfig(**lora_config)
                self.peft_config = lora_config
            # you cannot fine-tune quantized models without LoRA
            if quantization and not lora_config:
                lora_config = {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                }
                self.peft_config = LoraConfig(target_modules=peft_target_modules, **lora_config)
            if lora_config:
                self.log.info(f"LoRA config: {self.peft_config}")

            # Load model and tokenizer
            if quantization == 8:
                # Use AutoConfig to automatically load the configuration
                if self.model_name.lower() == "local":  # type: ignore
                    self.log.info(f"Loading local model {model_class} : {self.input.get()}")
                    if not self.config:  # type: ignore
                        self.config = AutoConfig.from_pretrained(os.path.join(self.input.get(), "/model"))
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        os.path.join(self.input.get(), "/model"),
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        load_in_8bit=True,
                        config=self.config,
                        **kwargs,
                    )
                else:
                    self.log.info(f"Loading from huggingface hub: {model_class} : {model_name}")
                    if not self.config:
                        self.config = AutoConfig.from_pretrained(self.model_name)
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        self.model_name,
                        revision=model_revision,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        load_in_8bit=True,
                        config=self.config,
                        **kwargs,
                    )
            elif quantization == 4:
                # Use AutoConfig to automatically load the configuration
                if self.model_name.lower() == "local":  # type: ignore
                    self.log.info(f"Loading local model {model_class} : {self.input.get()}")
                    if not self.config:
                        self.config = AutoConfig.from_pretrained(os.path.join(self.input.get(), "/model"))
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        os.path.join(self.input.get(), "/model"),
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        load_in_4bit=True,
                        config=self.config,
                        **kwargs,
                    )
                else:
                    self.log.info(f"Loading from huggingface hub: {model_class} : {model_name}")
                    if not self.config:
                        self.config = AutoConfig.from_pretrained(self.model_name)
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        self.model_name,
                        revision=model_revision,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        load_in_4bit=True,
                        config=self.config,
                        **kwargs,
                    )
            else:
                # Use AutoConfig to automatically load the configuration
                if self.model_name.lower() == "local":  # type: ignore
                    self.log.info(f"Loading local model {model_class} : {self.input.get()}")
                    if not self.config:
                        self.config = AutoConfig.from_pretrained(os.path.join(self.input.get(), "/model"))
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        os.path.join(self.input.get(), "/model"),
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        config=self.config,
                        **kwargs,
                    )
                else:
                    self.log.info(f"Loading from huggingface hub: {model_class} : {model_name}")
                    if not self.config:
                        self.config = AutoConfig.from_pretrained(self.model_name)
                    self.model = getattr(__import__("transformers"), str(model_class)).from_pretrained(
                        model_name,
                        revision=model_revision,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                        config=self.config,
                        **kwargs,
                    )

            if ":" in tokenizer_name:
                tokenizer_revision = tokenizer_name.split(":")[1]
                tokenizer_name = tokenizer_name.split(":")[0]
            else:
                tokenizer_revision = None
            self.tokenizer_name = tokenizer_name
            self.tokenizer_revision = tokenizer_revision

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

    def upload_to_hf_hub(
        self,
        hf_repo_id: Optional[str] = None,
        hf_commit_message: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_private: Optional[str] = None,
        hf_create_pr: Optional[str] = None,
    ):
        """Upload the model and tokenizer to Hugging Face Hub."""
        try:
            if self.model:
                self.model.to("cpu").push_to_hub(
                    repo_id=hf_repo_id if hf_repo_id else self.hf_repo_id,  # type: ignore
                    commit_message=hf_commit_message if hf_commit_message else self.hf_commit_message,  # type: ignore
                    token=hf_token if hf_token else self.hf_token,  # type: ignore
                    private=hf_private if hf_private else self.hf_private,  # type: ignore
                    create_pr=hf_create_pr if hf_create_pr else self.hf_create_pr,  # type: ignore
                )
            if self.tokenizer:
                self.tokenizer.push_to_hub(
                    repo_id=hf_repo_id if hf_repo_id else self.hf_repo_id,  # type: ignore
                    commit_message=hf_commit_message if hf_commit_message else self.hf_commit_message,  # type: ignore
                    token=hf_token if hf_token else self.hf_token,  # type: ignore
                    private=hf_private if hf_private else self.hf_private,  # type: ignore
                    create_pr=hf_create_pr if hf_create_pr else self.hf_create_pr,  # type: ignore
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
        precision: str = "bfloat16",
        quantization: Optional[int] = None,
        lora_config: Optional[dict] = None,
        use_accelerate: bool = False,
        use_trl: bool = False,
        accelerate_no_split_module_classes: List[str] = [],
        compile: bool = False,
        evaluate: bool = False,
        save_steps: int = 500,
        save_total_limit: Optional[int] = None,
        load_best_model_at_end: bool = False,
        metric_for_best_model: Optional[str] = None,
        greater_is_better: Optional[bool] = None,
        map_data: Optional[Callable] = None,
        use_huggingface_dataset: bool = False,
        huggingface_dataset: str = "",
        hf_repo_id: Optional[str] = None,
        hf_commit_message: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_private: bool = True,
        hf_create_pr: bool = False,
        notification_email: str = "",
        learning_rate: float = 1e-5,
        **kwargs,
    ):
        """
        Fine-tunes a pre-trained Hugging Face model.

        Args:
            model_name (str): The name of the pre-trained model.
            tokenizer_name (str): The name of the pre-trained tokenizer.
            num_train_epochs (int): The total number of training epochs to perform.
            per_device_batch_size (int): The batch size per device during training.
            model_class (str, optional): The model class to use. Defaults to "AutoModel".
            tokenizer_class (str, optional): The tokenizer class to use. Defaults to "AutoTokenizer".
            device_map (str | dict, optional): The device map for distributed training. Defaults to "auto".
            precision (str, optional): The precision to use for training. Defaults to "bfloat16".
            quantization (int, optional): The quantization level to use for training. Defaults to None.
            lora_config (dict, optional): Configuration for PEFT LoRA optimization. Defaults to None.
            use_accelerate (bool, optional): Whether to use accelerate for distributed training. Defaults to False.
            use_trl (bool, optional): Whether to use TRL for training. Defaults to False.
            accelerate_no_split_module_classes (List[str], optional): The module classes to not split during distributed training. Defaults to [].
            evaluate (bool, optional): Whether to evaluate the model after training. Defaults to False.
            compile (bool, optional): Whether to compile the model before fine-tuning. Defaults to True.
            save_steps (int, optional): Number of steps between checkpoints. Defaults to 500.
            save_total_limit (Optional[int], optional): Maximum number of checkpoints to keep. Older checkpoints are deleted. Defaults to None.
            load_best_model_at_end (bool, optional): Whether to load the best model (according to evaluation) at the end of training. Defaults to False.
            metric_for_best_model (Optional[str], optional): The metric to use to compare models. Defaults to None.
            greater_is_better (Optional[bool], optional): Whether a larger value of the metric indicates a better model. Defaults to None.
            use_huggingface_dataset (bool, optional): Whether to load a dataset from huggingface hub.
            huggingface_dataset (str, optional): The huggingface dataset to use.
            map_data (Callable, optional): A function to map data before training. Defaults to None.
            hf_repo_id (str, optional): The Hugging Face repo ID. Defaults to None.
            hf_commit_message (str, optional): The Hugging Face commit message. Defaults to None.
            hf_token (str, optional): The Hugging Face token. Defaults to None.
            hf_private (bool, optional): Whether to make the repo private. Defaults to True.
            hf_create_pr (bool, optional): Whether to create a pull request. Defaults to False.
            notification_email (str, optional): Whether to notify after job is complete. Defaults to None.
            learning_rate (float, optional): Learning rate for backpropagation.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            None
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
            self.precision = precision
            self.quantization = quantization
            self.lora_config = lora_config  # type: ignore
            self.use_accelerate = use_accelerate
            self.use_trl = use_trl
            self.accelerate_no_split_module_classes = accelerate_no_split_module_classes
            self.evaluate = evaluate
            self.use_huggingface_dataset = use_huggingface_dataset
            self.huggingface_dataset = huggingface_dataset
            self.hf_repo_id = hf_repo_id
            self.hf_commit_message = hf_commit_message
            self.hf_token = hf_token
            self.hf_private = hf_private
            self.hf_create_pr = hf_create_pr
            self.map_data = map_data
            self.notification_email = notification_email
            self.learning_rate = learning_rate
            self.config = None

            model_kwargs = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}
            self.model_kwargs = model_kwargs

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
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

            # Load dataset
            dataset_kwargs = {k.replace("data_", ""): v for k, v in kwargs.items() if "data_" in k}
            self.dataset_kwargs = dataset_kwargs
            self.preprocess_data(**dataset_kwargs)

            # Separate training and evaluation arguments
            trainer_kwargs = {k.replace("trainer_", ""): v for k, v in kwargs.items() if "trainer_" in k}
            self.trainer_kwargs = trainer_kwargs
            training_kwargs = {k.replace("training_", ""): v for k, v in kwargs.items() if "training_" in k}
            self.training_kwargs = training_kwargs

            # Create training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.output.output_folder, "model"),
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_batch_size,
                per_device_eval_batch_size=per_device_batch_size,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=greater_is_better,
                dataloader_num_workers=4,
                learning_rate=self.learning_rate,
                **training_kwargs,
            )

            # Add adapters to the model for fine-tuning
            if self.lora_config and not use_trl:
                self.model.enable_input_require_grads()
                self.model = get_peft_model(self.model, peft_config=self.peft_config)

            if compile:
                self.model = torch.compile(self.model)

            # Create trainer
            if use_trl:
                self.model = get_peft_model(self.model, peft_config=self.peft_config)
                trainer = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metrics,
                    data_collator=self.data_collator if hasattr(self, "data_collator") else None,
                    peft_config=self.peft_config,
                    **trainer_kwargs,
                )
            else:
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metrics,
                    data_collator=self.data_collator if hasattr(self, "data_collator") else None,
                    **trainer_kwargs,
                )

            # Train the model
            trainer.train()
            trainer.save_model(self.output.output_folder)

            if self.evaluate:
                eval_result = trainer.evaluate()
                self.log.info(f"Evaluation results: {eval_result}")

            # Save the model configuration to Hugging Face Hub if hf_repo_id is not None
            if self.hf_repo_id and self.config:
                self.config.save_pretrained(os.path.join(self.output.output_folder, "model"))
                self.upload_to_hf_hub()
        except Exception as e:
            self.log.exception(f"Failed to fine tune model: {e}")
            self.state.set_state(self.id, {"success": False, "exception": str(e)})
            raise
        self.state.set_state(self.id, {"success": True})

        self.done()

    def done(self):
        if self.notification_email:
            self.output.flush()
            send_fine_tuning_email(
                recipient=self.notification_email, bucket_name=self.output.bucket, prefix=self.output.s3_folder
            )
