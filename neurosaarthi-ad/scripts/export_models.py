"""Export fitted NeuroSaarthi-AD models to disk for Hugging Face upload."""

import joblib
import json
from pathlib import Path

# ── Import your demo runtime which trains the models on synthetic data ──
from demo.runtime import NeuroSaarthiRuntime

OUT_DIR = Path("hf_upload")
OUT_DIR.mkdir(exist_ok=True)

# 1. Build the runtime (trains all sub-models on synthetic data)
runtime = NeuroSaarthiRuntime.build()

# 2. Save classification pipelines (one per horizon)
for horizon, pipeline in runtime.classifiers.items():
    joblib.dump(pipeline, OUT_DIR / f"classifier_{horizon}yr.joblib")

# 3. Save the cognitive-trajectory regressor
joblib.dump(runtime.progression, OUT_DIR / f"progression_regressor.joblib")

# 4. Save the twin-lite retrieval index
joblib.dump(runtime.twinlite, OUT_DIR / f"twinlite_retriever.joblib")

# 5. Save config / feature metadata
config = {
    "horizons": list(runtime.classifiers.keys()),
    "modality_features": {
        "Cognition + clinical": [
            "age", "education_years", "sex_binary", "rural_indicator",
            "cognitive_score", "memory_score", "executive_score",
        ],
        "MRI": ["hippocampal_volume_mm3", "wmh_burden_ml"],
        "Blood": ["hba1c_percent", "hs_crp_mg_l"],
        "OCT/OCTA": ["rnfl_um", "vessel_density_percent"],
        "Genomics": ["apoe_e4_count", "ancestry_pc1"],
    },
    "modality_weights": {
        "Cognition + clinical": 0.40,
        "MRI": 0.23,
        "Blood": 0.14,
        "OCT/OCTA": 0.11,
        "Genomics": 0.12,
    },
    "framework": "scikit-learn",
    "python_requires": ">=3.10",
}
with open(OUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f" All artifacts saved to {OUT_DIR.resolve()}")
