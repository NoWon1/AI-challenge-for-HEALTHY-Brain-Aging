import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass
class ParticipantBaseline:
    sex: str
    sex_binary: int
    age: float
    education: float
    apoe_e4: int
    ancestry_pc1: float
    vulnerability: float
    baseline_cognition: float
    hippocampal_volume: float
    wmh_burden: float
    hba1c: float
    hs_crp: float
    rnfl: float
    vessel_density: float
    memory_score: float
    executive_score: float
    entorhinal_thickness: float
    ventricular_volume: float
    cortical_thickness_mean: float
    total_cholesterol: float
    fasting_glucose: float
    gfaz_area: float
    annual_hazard: float
    event: int
    event_time_years: float
    event_time_days: int
