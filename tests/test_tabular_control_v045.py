from ids.models.registry import get_model_builder


def test_mlp_control_registered_without_tensorflow_construction():
    assert get_model_builder("mlp_canonical").__name__ == "build_mlp_binary"


def test_mlp_dann_control_registered_without_tensorflow_construction():
    assert get_model_builder("mlp_dann_canonical").__name__ == "build_mlp_dann"
