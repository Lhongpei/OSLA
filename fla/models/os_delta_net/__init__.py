# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.os_delta_net.configuration_os_delta_net import OSDNConfig
from fla.models.os_delta_net.modeling_os_delta_net import OSDNForCausalLM, OSDNModel

AutoConfig.register(OSDNConfig.model_type, OSDNConfig, exist_ok=True)
AutoModel.register(OSDNConfig, OSDNModel, exist_ok=True)
AutoModelForCausalLM.register(OSDNConfig, OSDNForCausalLM, exist_ok=True)

__all__ = ['OSDNConfig', 'OSDNForCausalLM', 'OSDNModel']
