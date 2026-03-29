# -*- coding: utf-8 -*-

from .chunk import chunk_osla_delta_rule
from .fused_recurrent import fused_recurrent_delta_rule

__all__ = [
    'chunk_osla_delta_rule',
    'fused_recurrent_delta_rule',
]
