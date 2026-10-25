"""GSXA: Gated Scalable Exclusive Self Attention, in a BERT-style encoder
(inference code).

Importing the package registers the classes with the ``transformers`` Auto
factories, and marks them for ``register_for_auto_class`` so ``save_pretrained``
writes an ``auto_map`` and copies this code next to the weights (load such a
directory or Hub repo with ``trust_remote_code=True``).
"""

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

from .configuration_gsxa import ATTENTION_TYPES, GSXAConfig
from .modeling_gsxa import (
    GSXAForMaskedLM,
    GSXAForSequenceClassification,
    GSXAModel,
    GSXAPreTrainedModel,
)

__version__ = "0.1.0"


def _register():
    try:
        AutoConfig.register(GSXAConfig.model_type, GSXAConfig)
    except ValueError:  # already registered (re-import)
        return
    AutoModel.register(GSXAConfig, GSXAModel)
    AutoModelForMaskedLM.register(GSXAConfig, GSXAForMaskedLM)
    AutoModelForSequenceClassification.register(GSXAConfig, GSXAForSequenceClassification)
    GSXAConfig.register_for_auto_class()
    GSXAModel.register_for_auto_class("AutoModel")
    GSXAForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
    GSXAForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")


_register()

__all__ = [
    "ATTENTION_TYPES",
    "GSXAConfig",
    "GSXAPreTrainedModel",
    "GSXAModel",
    "GSXAForMaskedLM",
    "GSXAForSequenceClassification",
    "__version__",
]
