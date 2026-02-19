import subprocess
import sys
from pathlib import Path

# Get the scripts directory
scripts_dir = Path(__file__).parent

# List of scripts to run in sequence
scripts = [
    "01_reference_table.py",
    "02_geospatial_operations.py",
    "03_agronomic_features.py",
    "04_economic_features.py",
    "05_water_features.py",
    "06_emissions_fuel.py",
    "07_emissions_inputs.py",
    "08_emissions_residues.py",
    "09_emissions_soc.py",
    "10_emissions_albedo.py",
    "11_emissions_net_ghg.py",
    "12_esi_scores.py",
]


def run_scripts():
    """Execute all scripts in sequence."""

    for i, script in enumerate(scripts, 1):
        script_path = scripts_dir / script
        print(f"\n{'=' * 60}")
        print(f"Running ({i}/{len(scripts)}): {script}")
        print(f"{'=' * 60}\n")

        result = subprocess.run([sys.executable, str(script_path)], cwd=scripts_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("PIPELINE EXECUTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"✓ All {len(scripts)} scripts executed!")


if __name__ == "__main__":
    run_scripts()
