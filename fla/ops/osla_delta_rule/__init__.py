# -*- coding: utf-8 -*-

from .chunk import chunk_osla_delta_rule
from .fused_recurrent import fused_recurrent_delta_rule
from .fused_recurrent_osgm import fused_recurrent_delta_rule_osgm
from .chunk_osgm import chunk_delta_rule_osgm
__all__ = [
    'chunk_osla_delta_rule',
    'chunk_delta_rule_osgm',
    'fused_recurrent_delta_rule',
    'fused_recurrent_delta_rule_osgm',
]
