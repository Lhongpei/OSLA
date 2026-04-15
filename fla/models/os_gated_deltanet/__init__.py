# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.os_gated_deltanet.configuration_os_gated_deltanet import OSGDNConfig
from fla.models.os_gated_deltanet.modeling_os_gated_deltanet import OSGDNForCausalLM, OSGDNModel

AutoConfig.register(OSGDNConfig.model_type, OSGDNConfig, exist_ok=True)
AutoModel.register(OSGDNConfig, OSGDNModel, exist_ok=True)
AutoModelForCausalLM.register(OSGDNConfig, OSGDNForCausalLM, exist_ok=True)

__all__ = ['OSGDNConfig', 'OSGDNForCausalLM', 'OSGDNModel']
