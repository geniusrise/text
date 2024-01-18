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

from .classification import TextClassificationAPI, TextClassificationBulk, TextClassificationFineTuner
from .embeddings import EmbeddingsAPI, EmbeddingsBulk
from .instruction import InstructionAPI, InstructionBulk, InstructionFineTuner
from .language_model import LanguageModelAPI, LanguageModelBulk, LanguageModelFineTuner
from .ner import NamedEntityRecognitionAPI, NamedEntityRecognitionBulk, NamedEntityRecognitionFineTuner
from .nli import NLIAPI, NLIBulk, NLIFineTuner
from .qa import QAAPI, QABulk, QAFineTuner
from .summarization import SummarizationAPI, SummarizationBulk, SummarizationFineTuner
from .translation import TranslationAPI, TranslationBulk, TranslationFineTuner
from .notebook import TextJupyterNotebook
