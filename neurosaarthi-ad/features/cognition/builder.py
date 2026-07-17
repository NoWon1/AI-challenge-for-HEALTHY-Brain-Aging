from features.base import pivot_modality_features


def build_cognition_features(features):
    return pivot_modality_features(features, "cognition")

