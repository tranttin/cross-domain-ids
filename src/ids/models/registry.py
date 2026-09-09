from __future__ import annotations

from ids.models.hybrid import build_legacy_binary_hybrid, build_legacy_multiclass_hybrid
from ids.models.dann import build_legacy_dann
from ids.models.tabular import build_mlp_binary, build_mlp_dann


BUILDERS = {
    "cnn_bilstm_attention_binary": build_legacy_binary_hybrid,
    "cnn_bilstm_attention_multiclass": build_legacy_multiclass_hybrid,
    "legacy_dann": build_legacy_dann,
    "mlp_canonical": build_mlp_binary,
    "mlp_dann_canonical": build_mlp_dann,
}


def get_model_builder(name: str):
    try:
        return BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(sorted(BUILDERS))}") from exc
