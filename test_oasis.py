import pandas as pd
from etl.oasis.adapter import OasisAdapter

# Dummy data generator
root = "data/raw/oasis"
import os
os.makedirs(root, exist_ok=True)
pd.DataFrame({
    'Subject': ['P-1', 'P-1', 'P-2'],
    'Age': [70, 71, 65],
    'dx1': ['cn', 'mci', 'normal'],
    'CDR': [0, 0.5, 0],
    'MMSE': [30, 28, 29]
}).to_csv(f"{root}/OASIS3_clinical.csv", index=False)

pd.DataFrame({
    'Subject': ['P-1', 'P-2'],
    'M/F': ['M', 'F'],
    'EDUC': [16, 12]
}).to_csv(f"{root}/OASIS3_demographics.csv", index=False)

adapter = OasisAdapter()
tables = adapter.extract()
print(tables.modality_features)
