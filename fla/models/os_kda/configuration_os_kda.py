# -*- coding: utf-8 -*-
"""OS-KDA config: Kimi Delta Attention with OSGM preconditioning."""

from fla.models.kda.configuration_kda import KDAConfig


class OSKDAConfig(KDAConfig):
    model_type = "os_kda"

    def __init__(
        self,
        use_osgm: bool = True,
        osgm_eta: float | None = 1.0,
        osgm_use_denominator: bool | None = False,
        osgm_d_min: float | None = 0.0,
        osgm_d_max: float | None = 1e9,
        osgm_beta_aware: bool = True,
        osgm_decay_mode: str = "constant",
        osgm_decay_gamma: float = 0.999,
        **kwargs,
    ):
        super().__init__(
            use_osgm=use_osgm,
            osgm_eta=osgm_eta,
            osgm_use_denominator=osgm_use_denominator,
            osgm_d_min=osgm_d_min,
            osgm_d_max=osgm_d_max,
            osgm_beta_aware=osgm_beta_aware,
            osgm_decay_mode=osgm_decay_mode,
            osgm_decay_gamma=osgm_decay_gamma,
            **kwargs,
        )
