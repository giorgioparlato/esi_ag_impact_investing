# ESI-CIMMYT

This repository contains the Python pipeline used to calculate Earth
System Impact (ESI) scores from CIMMYT maize production data. The
pipeline combines agronomic, economic, water, and geospatial information;
estimates greenhouse-gas (GHG) components; and calculates absolute and
intensity-based ESI scores.

The main unit of analysis is `practice_id`, which identifies unique plots
of cultivated land. Final ESI scores are calculated for practices
classified as maize only.

## Repository Structure

```text
code/
  esi_computation/
    scripts/          Ordered pipeline scripts and shared path configuration
environment.yml       Conda environment definition
README.md             Setup and execution guide
```

Input data, generated outputs, exploratory notebooks, and local environment
files are excluded from version control.

## Environment Setup

The pipeline uses Python 3.13 and the dependencies declared in
`environment.yml`.

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate env_esi_cimmyt
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
```

## Data Setup

The input data are not distributed with this repository. The data root must
contain the expected `raw` inputs and is also used for generated `processed`
outputs. The pipeline creates the processed output directories when needed. The
main directory structure is:

```text
data/
  raw/
    cimmyt/
      dataverse_files/
    esi_coefficients/
    geospatial/
      raster/
      vector/
    water/
  processed/
    tables/
    sql_db/
```

The input filenames and subdirectories are defined in
`code/esi_computation/scripts/paths.py`.

### Configure the Data Root

By default, the pipeline uses the `data` directory at the top level of the
repository. A different location can be supplied through the
`ESI_CIMMYT_DATA` environment variable. The selected directory must contain the
`raw` folder shown above; the `processed` folder is created automatically.

PowerShell:

```powershell
$env:ESI_CIMMYT_DATA = "C:\path\to\data"
```

macOS or Linux:

```bash
export ESI_CIMMYT_DATA="/path/to/data"
```

Run the pipeline from the same terminal session (e.g. on VS Code)
in which the variable was set.

## Running the Pipeline

From the repository root, with the Conda environment activated, run:

```bash
python code/esi_computation/scripts/00_run_all.py
```

The runner executes 13 processing scripts in a fixed order using the active
Python environment. It stops at the first script that returns an error. Running
the pipeline again replaces outputs that use the same filenames and SQLite
table names.

Individual scripts can also be run directly, but their required upstream
tables must already exist and be current.

## Pipeline Stages

| Script | Purpose |
|---|---|
| `01_reference_table.py` | Creates the practice-level reference table. |
| `02_geospatial_operations.py` | Adds geospatial classifications and ESI coefficients. |
| `03_agronomic_features.py` | Calculates agronomic features and yield-unit support statuses. |
| `04_economic_features.py` | Calculates costs, revenues, and related economic features. |
| `05_water_features.py` | Calculates water-use variables and irrigation emissions. |
| `06_emissions_fuel.py` | Calculates fuel emissions and fuel-data statuses. |
| `07_emissions_inputs.py` | Calculates synthetic- and organic-input emissions. |
| `08_emissions_residues.py` | Calculates crop-residue emissions and completeness statuses. |
| `09_emissions_soc.py` | Calculates soil organic carbon change and assumption statuses. |
| `10_emissions_albedo.py` | Calculates emissions-equivalent albedo effects. |
| `11_emissions_net_ghg.py` | Calculates net GHG emissions and invalidation reasons. |
| `12_esi_scores.py` | Calculates component, total, and intensity ESI scores. |

## Outputs

Each processing stage writes a Feather table under:

```text
<data root>/processed/tables/
```

Corresponding tables are also written to:

```text
<data root>/processed/sql_db/cimmyt.db
```

The processed tables are numbered according to the pipeline stages. The final
ESI output is available as `table_12_esi_scores.feather` and as the
`table_12_esi_scores` table in SQLite.

## Missing Data and Quality Flags

Records are retained when information is incomplete. Diagnostic and operational
flags document whether values were reported, imputed, assumed, missing,
unsupported, or not applicable. Operational completeness rules can set
component emissions, net GHG estimates, or ESI scores to missing when required
information is unavailable.

The processed module tables contain the detailed flags relevant to each stage.
In particular, `table_11_emissions_net_ghg` records reasons for incomplete net
GHG estimates, while `table_12_esi_scores` records the availability and
calculation status of ESI component and total scores. Scientific definitions,
formulas, emission factors, assumptions, and full flag interpretations are
maintained in the separate methodology documentation accompanying
accompanying the main scientific publication.
