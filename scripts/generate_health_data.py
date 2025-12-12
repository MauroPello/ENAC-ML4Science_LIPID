# TODO: Add docstrings
# TODO: Add customization options for the distributions of the data (e.g., age distribution range, health condition prevalence, etc.)

import os
import sys
from pathlib import Path
import pandas
import numpy as np
import random
import matplotlib.pyplot as plt

root = Path(__file__).parent.parent
sys.path.append(str(root))


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)


def _get_neighborhoods_ids(path: Path) -> np.ndarray:
    """Read neighborhood IDs from the morphology data file.

    Args:
        path (Path): Path to the morphology data CSV.

    Returns:
        np.ndarray: Array of unique neighborhood IDs.
    """
    morphology_data = pandas.read_csv(path)
    neighborhoods_ids = morphology_data["id"].unique()
    return neighborhoods_ids


def generate_health_data(
    root: Path = root,
    morphology_data_path: Path = root / "data" / "morphology_data_cleaned.csv",
    participants_per_neighborhood_range: tuple = (8, 15),
    output_path: Path = root / "data" / "synthetic_health_data.xlsx",
) -> None:
    """Generate synthetic health data linked to neighborhood morphology.

    Args:
        root (Path): Root directory of the project.
        morphology_data_path (Path): Path to the cleaned morphology data.
        participants_per_neighborhood_range (tuple): Min and max participants per neighborhood.
        output_path (Path): Path to save the generated excel file.
    """

    # Reproducibility
    set_seed(42)

    # Participants and Neighborhoods
    neighborhoods_ids = _get_neighborhoods_ids(morphology_data_path)
    n_neighborhoods = len(neighborhoods_ids)
    n_participants_per_neighborhood = np.random.randint(
        participants_per_neighborhood_range[0],
        participants_per_neighborhood_range[1],
        size=n_neighborhoods,
    )

    # Total number of participants
    n_participants = np.sum(n_participants_per_neighborhood)

    # Participant IDs
    participant_ids = np.arange(1, n_participants + 1)
    assert len(participant_ids) == n_participants

    # Neighborhood IDs
    neighborhood_ids = []
    for i, n in enumerate(n_participants_per_neighborhood):
        neighborhood_id = neighborhoods_ids[i]
        neighborhood_ids.extend([neighborhood_id] * n)
    neighborhood_ids = np.array(neighborhood_ids)
    assert len(neighborhood_ids) == n_participants

    # Socio-Demographic Data
    ages = np.random.randint(18, 80, size=n_participants)
    sexes = np.random.choice(["Male", "Female"], size=n_participants)
    incomes = np.random.choice(["Low", "Medium", "High"], size=n_participants)
    education_levels = np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"], size=n_participants
    )
    socio_demograph_data = pandas.DataFrame(
        {
            "participant_id": participant_ids,
            "neighborhood_id": neighborhood_ids,
            "age": ages,
            "sex": sexes,
            "income": incomes,
            "education_level": education_levels,
        }
    )

    # Health Data
    heart_failures = np.random.choice([0, 1], size=n_participants, p=[0.9, 0.1])
    heart_rhythms = np.random.choice([0, 1], size=n_participants, p=[0.85, 0.15])
    d_metabolic_diabetes_I = np.random.choice(
        [0, 1], size=n_participants, p=[0.95, 0.05]
    )
    d_metabolic_diabetes_II = np.random.choice(
        [0, 1], size=n_participants, p=[0.9, 0.1]
    )
    d_metabolic_obesity = np.random.choice([0, 1], size=n_participants, p=[0.8, 0.2])
    d_breath_respiratory = np.random.choice([0, 1], size=n_participants, p=[0.85, 0.15])
    d_breath_asthma = np.random.choice([0, 1], size=n_participants, p=[0.9, 0.1])
    GHQ12_scores = np.random.randint(0, 13, size=n_participants)
    points_sleep_deprivation = np.random.randint(0, 21, size=n_participants)
    sleep_disorder_hot = np.random.choice([0, 1], size=n_participants, p=[0.8, 0.2])
    sleeping_hours = np.random.uniform(4, 10, size=n_participants)
    bedtime_hours = [
        f"{random.randint(20, 23)}:{random.randint(0, 59):02d}"
        for _ in range(n_participants)
    ]
    health_data = pandas.DataFrame(
        {
            "participant_id": participant_ids,
            "neighborhood_id": neighborhood_ids,
            "heart_failure": heart_failures,
            "heart_rhythm": heart_rhythms,
            "d_metabolic_diabetes_I": d_metabolic_diabetes_I,
            "d_metabolic_diabetes_II": d_metabolic_diabetes_II,
            "d_metabolic_obesity": d_metabolic_obesity,
            "d_breath_respiratory": d_breath_respiratory,
            "d_breath_asthma": d_breath_asthma,
            "GHQ12_score": GHQ12_scores,
            "points_sleep_deprivation": points_sleep_deprivation,
            "sleep_disorder_hot": sleep_disorder_hot,
            "sleeping_hours": sleeping_hours,
            "bedtime_hour": bedtime_hours,
        }
    )

    # Save the data to XLSX file, with two sheets: "Participant_SocioDemograph_Data" and "Participant_HEALTH_Data"
    with pandas.ExcelWriter(output_path) as writer:
        socio_demograph_data.to_excel(
            writer, sheet_name="Participant_SocioDemograph_Data", index=False
        )
        health_data.to_excel(writer, sheet_name="Participant_HEALTH_Data", index=False)

    print(
        f"Synthetic health data generated with {n_participants} rows and saved to {output_path}"
    )


if __name__ == "__main__":
    # TODO: Add arguments for CLI customization
    generate_health_data()
