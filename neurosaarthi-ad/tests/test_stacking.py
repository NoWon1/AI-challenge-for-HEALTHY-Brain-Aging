import numpy as np

from models.fusion.stacking import StackedFusionModel


def _make_predictions(n=50, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "model_a": rng.uniform(0, 1, n),
        "model_b": rng.uniform(0, 1, n),
    }


def test_stacking_fit_predict():
    preds = _make_predictions()
    y_true = np.random.randint(0, 2, 50)
    pids = np.array([f"P{i}" for i in range(50)])

    model = StackedFusionModel(base_model_names=["model_a", "model_b"])
    model.fit(preds, y_true, pids)

    fused = model.predict(preds)
    assert len(fused) == 50
    assert (fused >= 0).all()
    assert (fused <= 1).all()


def test_stacking_learned_weights():
    preds = _make_predictions()
    y_true = np.random.randint(0, 2, 50)
    pids = np.array([f"P{i}" for i in range(50)])

    model = StackedFusionModel(base_model_names=["model_a", "model_b"])
    model.fit(preds, y_true, pids)

    weights = model.learned_weights()
    assert "model_a" in weights
    assert "model_b" in weights


def test_stacking_handles_nan():
    preds = _make_predictions()
    preds["model_a"][0] = np.nan
    y_true = np.random.randint(0, 2, 50)
    pids = np.array([f"P{i}" for i in range(50)])

    model = StackedFusionModel(base_model_names=["model_a", "model_b"])
    model.fit(preds, y_true, pids)

    fused = model.predict(preds)
    assert not fused.isna().any()
