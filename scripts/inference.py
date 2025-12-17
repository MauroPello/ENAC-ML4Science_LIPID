from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.prediction import infer_neighborhood_health_risks


# you can move your own csv to the data folder, change the path here
# and run inference on the neighborhood data in your csv

# Debug diagnostics
csv_path = project_root / "data" / "morphology_data_cleaned.csv"
outputs_dir = project_root / "outputs"
health_risks = infer_neighborhood_health_risks(
    morph_csv_path = csv_path,
    outputs_dir = outputs_dir,
)

for name, df_risk in health_risks.items():
    print(f"\n{name.upper()} risk table")
    print(df_risk)

# Average health risk per typology across all socio-demographic combinations (wide view)

rows = []
for name, df_risk in health_risks.items():
    risk_col = f"risk_{name}"
    if risk_col not in df_risk.columns:
        print(f"Skipping {name}: column '{risk_col}' not found.")
        continue
    grouped = (
        df_risk.groupby("typology")[risk_col]
        .mean()
        .reset_index()
        .rename(columns={risk_col: name})
    )
    rows.append(grouped)

if rows:
    combined = rows[0]
    for g in rows[1:]:
        combined = combined.merge(g, on="typology", how="outer")
    combined = combined.sort_values("typology")
    print("Average risk by typology (one column per dataset)")
    print(combined)
else:
    print("No risk data available to summarize.")
    sys.exit(0)

# Flag typologies with elevated modeled risk per dataset (no melting)

risk_cols = [c for c in combined.columns if c != "typology"]
for col in risk_cols:
    df = combined[["typology", col]].rename(columns={col: "risk"})

    mean_risk = df["risk"].mean()
    std_risk = df["risk"].std()
    zscores = (df["risk"] - mean_risk) / std_risk if std_risk else float("nan")
    df = df.assign(zscore=zscores.round(2))

    high = df[df["zscore"] >= 1.5].sort_values("zscore", ascending=False)
    low = df[df["zscore"] <= -1.5].sort_values("zscore", ascending=True)

    print(f"\n{col.upper()} health risk")

    if not high.empty:
        print(f"\nTypologies with alarmingly higher {col} risk (z >= 1.5):")
        print(high[["typology", "risk", "zscore"]])

    if not low.empty:
        print(f"\nTypologies with exceptionally lower {col} risk (z <= -1.5):")
        print(low[["typology", "risk", "zscore"]])

    print("\nPer-typology risk with zscore: ")
    print(df.sort_values("risk", ascending=False)[["typology", "risk", "zscore"]])

