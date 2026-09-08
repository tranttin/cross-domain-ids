def test_v048_builders_and_shared_layer_names():
    import pytest
    tf = pytest.importorskip("tensorflow")
    from ids.models.tabular import build_mlp_bn_fair_binary, build_mlp_bn_fair_dann, shared_layer_weights, load_shared_layer_weights
    src = build_mlp_bn_fair_binary(12)
    d = build_mlp_bn_fair_dann(12, lambda_d=0.0)
    w = shared_layer_weights(src)
    assert {"bnfair_dense_1","bnfair_bn_1","bnfair_embedding","cls_output"}.issubset(w)
    load_shared_layer_weights(d,w,freeze_batchnorm=True)
    assert d.get_layer("bnfair_bn_1").trainable is False
    for name in w:
        a=src.get_layer(name).get_weights(); b=d.get_layer(name).get_weights()
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert (x==y).all()
