import os
import sys
from pathlib import Path
import pandas as pd

root = Path(__file__).parent.parent
os.chdir(root)
sys.path.append(str(root))


def clean_morphology_data(
    input_file: str = "data/morphology_data.xlsx",
    sheet_name: str = "morphology_EQ_Geneva",
    output_file: str = "data/morphology_data_cleaned.csv",
):
    # Load the morphology data
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    # Print the columns with unique values == len(df) or unique values <= 1
    cols_to_remove = []
    for col in df.columns:
        # ID is kept even if unique
        if col == "id":
            continue
        if df[col].nunique() == len(df) or df[col].nunique() <= 1:
            cols_to_remove.append(col)

    # Remove the columns with the "bin" prefix, as they are binned versions of other columns
    bin_columns = [col for col in df.columns if col.startswith("bin")]
    cols_to_remove.extend(bin_columns)

    df = df.drop(columns=cols_to_remove)
    print("Removed Columns: ", cols_to_remove)
    print()

    # Remove rows with any NaN values
    len_before = len(df)
    df = df.dropna()
    len_after = len(df)

    print(
        f"Cleaned DataFrame shape: {df.shape} (after removing {len(cols_to_remove)} columns and {len_before-len_after} row(s))"
    )
    print()

    # Save the cleaned dataframe to a new CSV (more readable format)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    # TODO: Add command line arguments to specify input/output files and sheet names
    clean_morphology_data()
