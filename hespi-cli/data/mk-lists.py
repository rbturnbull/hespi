import pandas as pd
from pathlib import Path


def clean_value(value):
    value = value.strip()
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value


work_dir = Path(__file__).parent
df = pd.read_csv(work_dir / "plants.csv")
for column in ["family", "genus", "species", "authority"]:
    values = df[~pd.isna(df[column])][column].unique()
    values = set(clean_value(value) for value in values)
    values = set(value for value in values if value)
    values = sorted(values)
    output_path = work_dir / f"{column}.txt"
    print(output_path)
    output_path.write_text("\n".join(values))
