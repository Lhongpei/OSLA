# -*- coding: utf-8 -*-
"""OS-KDA model wrappers.

The implementation reuses the KDA model stack; OS behavior is selected by
``OSKDAConfig.use_osgm`` and handled inside ``KimiDeltaAttention``.
"""

from fla.models.kda.modeling_kda import KDAForCausalLM, KDAModel, KDAPreTrainedModel
from fla.models.os_kda.configuration_os_kda import OSKDAConfig


class OSKDAPreTrainedModel(KDAPreTrainedModel):
    config_class = OSKDAConfig
    _no_split_modules = ["KDABlock"]


class OSKDAModel(KDAModel):
    config_class = OSKDAConfig


class OSKDAForCausalLM(KDAForCausalLM):
    config_class = OSKDAConfig
