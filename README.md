# ENAC ML4Science LIPID Project

## Setup

In order to properly set up the repository, download the `morphology_EQ_Geneva.xlsx` file from the Microsoft Teams Documents, then create a `data` directory and move that file into that folder.

### 0. Conda Environment (Optional)
Using `conda`, create a new environment and activate it.

```bash
conda env create -f environment.yml
conda activate mlenv
```

### 1. Requirements
Install the requirements doing:

```bash
# Install the requirements
pip install -r requirements.txt
```

## Guidelines

Before pushing the code to the repository, make sure to:
1. Not push the `data` directory
2. Document the steps properly
3. Use the bash command `pipreqs --force` to overwrite the `requirements.txt` file with new libraries
4. In general, always check the modified files with `git status` before commits
 
## Project Structure

The complete outline for this project should look like:

```bash
.
├── data
│   ├── synthetic_health_data.xlsx              # Created by running ./scripts/generate_health_data.py
│   ├── morphology_data.xlsx                    # Original Data, downloaded from Microsoft Teams
│   └── morphology_data_cleaned.csv             # Briefly processed data, by running ./scripts/clean_morph_data.py
├── docs
│   ├── morph_data.md
│   └── health_data.md
├── notebooks
│   ├── analysis.ipynb
│   └── generate_health_data.ipynb
├── scripts
│   ├── clean_morph_data.py
│   └── generate_health_data.py
├── src
│   ├── model
│   │   └── ...
│   └── processing
│       └── ...
│
├── README.md
├── .gitignore
├── environment.yaml
└── requirements.txt
```