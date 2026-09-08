from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fair_builders_are_present_and_share_task_names():
    text = (ROOT / "src/ids/models/tabular.py").read_text(encoding="utf-8")
    assert "def build_mlp_fair_binary" in text
    assert "def build_mlp_fair_dann" in text
    assert "fair_embedding" in text
    assert "fair_cls_hidden" in text
    assert "LayerNormalization" in text


def test_runner_exposes_mlp_fair_and_matched_lr_default():
    text = (ROOT / "scripts/run_transfer_clean.py").read_text(encoding="utf-8")
    assert '"mlp_fair"' in text
    assert "effective_dann_lr" in text
    assert 'args.source_lr if args.encoder in {"mlp", "mlp_fair", "mlp_bn_fair"}' in text


def test_clean_dann_uses_joint_forward_pass():
    text = (ROOT / "src/ids/models/clean_dann.py").read_text(encoding="utf-8")
    assert "x_joint = tf.concat([xs, xt], axis=0)" in text
    assert "cls_joint, dom_joint = model(x_joint, training=True)" in text
