# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.os_kda.configuration_os_kda import OSKDAConfig
from fla.models.os_kda.modeling_os_kda import OSKDAForCausalLM, OSKDAModel

AutoConfig.register(OSKDAConfig.model_type, OSKDAConfig, exist_ok=True)
AutoModel.register(OSKDAConfig, OSKDAModel, exist_ok=True)
AutoModelForCausalLM.register(OSKDAConfig, OSKDAForCausalLM, exist_ok=True)

__all__ = ["OSKDAConfig", "OSKDAForCausalLM", "OSKDAModel"]
