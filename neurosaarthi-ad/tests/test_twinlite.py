import pandas as pd

from models.twinlite.retrieval import TwinLiteRetriever


def test_twinlite_retrieves_nearest_without_self():
    frame = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p3"],
            "age": [60.0, 61.0, 80.0],
            "memory_score": [20.0, 19.5, 5.0],
        }
    )
    retriever = TwinLiteRetriever(["age", "memory_score"]).fit(frame)
    result = retriever.query(frame.iloc[0], k=1, exclude_participant_id="p1")
    assert result.loc[0, "participant_id"] == "p2"

