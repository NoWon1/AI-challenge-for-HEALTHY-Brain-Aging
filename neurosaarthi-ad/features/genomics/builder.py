from features.base import pivot_modality_features


def build_genomics_features(features):
    return pivot_modality_features(features, "genomics")

