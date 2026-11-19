"""Compatibility shims for running TRL 0.26 against vLLM 0.26, importable before TRL."""

import importlib.util
import sys
import types


def apply_vllm_compat_shims() -> None:
    if importlib.util.find_spec("vllm") is None:
        return

    import vllm.sampling_params as vllm_sampling_params
    import vllm.transformers_utils
    import vllm.utils as vllm_utils

    stub_name = "vllm.transformers_utils.tokenizer"
    if stub_name not in sys.modules and importlib.util.find_spec(stub_name) is None:
        stub = types.ModuleType(stub_name)
        stub.__spec__ = importlib.util.spec_from_loader(stub_name, loader=None)
        from vllm.tokenizers.hf import get_cached_tokenizer

        stub.get_cached_tokenizer = get_cached_tokenizer
        sys.modules[stub_name] = stub
        vllm.transformers_utils.tokenizer = stub

    if not hasattr(vllm_sampling_params, "GuidedDecodingParams"):
        vllm_sampling_params.GuidedDecodingParams = vllm_sampling_params.StructuredOutputsParams

    if not hasattr(vllm_utils, "get_open_port"):
        from vllm.utils.network_utils import get_open_port

        vllm_utils.get_open_port = get_open_port

    import trl.import_utils as trl_import_utils

    if isinstance(trl_import_utils._vllm_ascend_available, tuple):
        trl_import_utils._vllm_ascend_available = bool(trl_import_utils._vllm_ascend_available[0])
