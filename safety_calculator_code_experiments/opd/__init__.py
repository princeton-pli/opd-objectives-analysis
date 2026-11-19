from vllm_compat import apply_vllm_compat_shims

apply_vllm_compat_shims()

from .opd_config import OPDConfig
from .opd_trainer import OPDTrainer


__all__ = ["OPDConfig", "OPDTrainer"]
