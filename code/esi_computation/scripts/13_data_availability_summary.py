"""
Summarize data availability and quality flags from completed pipeline outputs.

Main operations:
    - read practice-level outputs from the pipeline SQLite database
    - count available and unavailable estimates within explicit cohorts
    - summarize exact flag statuses and non-exclusive flag reasons
    - export the summaries as an Excel workbook and CSV files

Author: Dario Marone
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from datetime import datetime, timezone
from pathlib import Path
import platform
import sqlite3
import subprocess

import pandas as pd
from openpyxl.styles import Font, PatternFill

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
from paths import (
    PATH_DATA,
    PATH_ESI_COEFFICIENTS,
    PATH_PROCESSED_SUMMARIES,
    PATH_SQL_DB,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ESI_SCORE_COLUMNS = [
    "esi_land",
    "esi_water",
    "esi_climate",
    "esi_climate_wfuel",
    "esi_climate_albedo",
    "esi_total",
    "esi_total_wfuel",
    "total_esi_intensity_area",
    "total_esi_intensity_revenue",
    "total_wfuel_esi_intensity_area",
    "total_wfuel_esi_intensity_revenue",
    "land_esi_intensity_area",
    "land_esi_intensity_revenue",
    "water_esi_intensity_area",
    "water_esi_intensity_revenue",
    "climate_esi_intensity_area",
    "climate_esi_intensity_revenue",
    "climate_wfuel_esi_intensity_area",
    "climate_wfuel_esi_intensity_revenue",
    "climate_albedo_esi_intensity_area",
    "climate_albedo_esi_intensity_revenue",
]

ESI_SCORE_LABELS = {
    "esi_land": "Land ESI score",
    "esi_water": "Water ESI score",
    "esi_climate": "Climate ESI score excluding fuel",
    "esi_climate_wfuel": "Climate ESI score including fuel",
    "esi_climate_albedo": "Albedo ESI score",
    "esi_total": "Total ESI score excluding fuel",
    "esi_total_wfuel": "Total ESI score including fuel",
    "total_esi_intensity_area": "Total ESI intensity by area excluding fuel",
    "total_esi_intensity_revenue": (
        "Total ESI intensity by revenue excluding fuel"
    ),
    "total_wfuel_esi_intensity_area": (
        "Total ESI intensity by area including fuel"
    ),
    "total_wfuel_esi_intensity_revenue": (
        "Total ESI intensity by revenue including fuel"
    ),
    "land_esi_intensity_area": "Land ESI intensity by area",
    "land_esi_intensity_revenue": "Land ESI intensity by revenue",
    "water_esi_intensity_area": "Water ESI intensity by area",
    "water_esi_intensity_revenue": "Water ESI intensity by revenue",
    "climate_esi_intensity_area": "Climate ESI intensity by area excluding fuel",
    "climate_esi_intensity_revenue": (
        "Climate ESI intensity by revenue excluding fuel"
    ),
    "climate_wfuel_esi_intensity_area": (
        "Climate ESI intensity by area including fuel"
    ),
    "climate_wfuel_esi_intensity_revenue": (
        "Climate ESI intensity by revenue including fuel"
    ),
    "climate_albedo_esi_intensity_area": "Albedo ESI intensity by area",
    "climate_albedo_esi_intensity_revenue": "Albedo ESI intensity by revenue",
}

FLAG_COLUMNS = {
    "Agronomic": (
        "table_03_agronomic_features",
        ["unit_support_status_grain", "unit_support_status_forage"],
    ),
    "Water": (
        "table_05_b_water_features",
        ["water_consumption_flag", "water_use_source"],
    ),
    "Fuel": (
        "table_06_emissions_fuel",
        ["fuel_cons_status", "power_source_availability"],
    ),
    "Inputs": (
        "table_07_emissions_ag_inputs",
        [
            "flag_imputed_product_type",
            "flag_imputed_nitrogen",
            "flag_imputed_moisture_level",
            "flag_missing_emissions_syn_prod",
            "flag_missing_emissions_syn_tran",
            "flag_missing_emissions_syn_app",
            "flag_missing_emissions_org_app",
            "flag_imputed_amount",
            "flag_pesticides_use",
        ],
    ),
    "Residues": (
        "table_08_emissions_residues",
        ["flag_total_pct", "flag_burned_pct", "flag_residue_emissions"],
    ),
    "SOC": (
        "table_09_emissions_soc",
        ["flag_soc_emissions", "flag_soc_assumptions"],
    ),
    "Net GHG": (
        "table_11_emissions_net_ghg",
        ["flag_net_ghg", "flag_net_ghg_wfuel"],
    ),
    "ESI scores": (
        "table_12_esi_scores",
        [
            "flag_esi_land",
            "flag_esi_water",
            "flag_esi_climate",
            "flag_esi_climate_wfuel",
            "flag_esi_climate_albedo",
            "flag_esi_total",
            "flag_esi_total_wfuel",
            "flag_esi_intensity_area",
            "flag_esi_intensity_revenue",
        ],
    ),
}

TABLE_COLUMNS = {
    "table_01_reference": ["practice_id"],
    "table_03_agronomic_features": [
        "practice_id",
        "crops",
        "unit_support_status_grain",
        "unit_support_status_forage",
    ],
    "table_05_b_water_features": [
        "practice_id",
        "water_cons_hyb_ml_dam3",
        "water_consumption_flag",
        "water_use_source",
    ],
    "table_06_emissions_fuel": [
        "practice_id",
        "fuel_cons_tco2eq",
        "fuel_cons_status",
        "power_source_availability",
    ],
    "table_07_emissions_ag_inputs": [
        "practice_id",
        *FLAG_COLUMNS["Inputs"][1],
    ],
    "table_08_emissions_residues": [
        "practice_id",
        *FLAG_COLUMNS["Residues"][1],
    ],
    "table_09_emissions_soc": [
        "practice_id",
        "soc_delta_tco2eq",
        "flag_soc_emissions",
        "flag_soc_assumptions",
    ],
    "table_10_emissions_albedo": [
        "practice_id",
        "albedo_emissions_tco2eq",
    ],
    "table_11_emissions_net_ghg": [
        "practice_id",
        "net_ghg_emissions_tco2eq",
        "net_ghg_emissions_wfuel_tco2eq",
        "flag_net_ghg",
        "flag_net_ghg_wfuel",
    ],
    "table_12_esi_scores": [
        "practice_id",
        "crops",
        *ESI_SCORE_COLUMNS,
        *FLAG_COLUMNS["ESI scores"][1],
    ],
}

MODULE_ORDER = {
    "Reference": 1,
    "Agronomic": 3,
    "Water": 5,
    "Fuel": 6,
    "Inputs": 7,
    "Residues": 8,
    "SOC": 9,
    "Albedo": 10,
    "Net GHG": 11,
    "ESI scores": 12,
}

SYNTHETIC_INPUT_INCOMPLETE_PATTERN = (
    r"\bMISSING EMISSION FACTOR\b"
    r"|MISSING OR UNSUPPORTED UNIT"
    r"|\bUNIT CONVERSION\b"
)
COMPLETE_ORGANIC_STATUSES = {
    "AVAILABLE",
    "NOT APPLICABLE",
    "AVAILABLE | NOT APPLICABLE",
}
COMPLETE_RESIDUE_STATUSES = {"AVAILABLE", "NOT APPLICABLE"}


# -----------------------------------------------------------------------------
# Data import and validation
# -----------------------------------------------------------------------------
def quote_identifier(identifier: str) -> str:
    """Quote a known SQL identifier after validating its characters."""
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    return f'"{identifier}"'


def read_pipeline_tables(database_path: Path) -> dict[str, pd.DataFrame]:
    """Read required columns and fail clearly when pipeline outputs are stale."""
    frames = {}
    with sqlite3.connect(database_path) as con:
        existing_tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = set(TABLE_COLUMNS) - existing_tables
        if missing_tables:
            missing = ", ".join(sorted(missing_tables))
            raise RuntimeError(f"Required pipeline tables are missing: {missing}")

        for table, columns in TABLE_COLUMNS.items():
            table_sql = quote_identifier(table)
            available_columns = {
                row[1] for row in con.execute(f"PRAGMA table_info({table_sql})")
            }
            missing_columns = set(columns) - available_columns
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise RuntimeError(
                    f"{table} is missing current pipeline columns: {missing}. "
                    "Rerun the pipeline before generating the summary."
                )

            columns_sql = ", ".join(quote_identifier(col) for col in columns)
            frames[table] = pd.read_sql_query(
                f"SELECT {columns_sql} FROM {table_sql}", con
            )

    for table, frame in frames.items():
        if frame["practice_id"].duplicated().any():
            duplicate_count = frame["practice_id"].duplicated(keep=False).sum()
            raise RuntimeError(
                f"{table} contains {duplicate_count:,} rows with duplicate "
                "practice_id values."
            )

    return frames


def restrict_to_reference(
    frame: pd.DataFrame, reference_ids: pd.Index
) -> pd.DataFrame:
    """Restrict a module table to practices in the reference table."""
    return frame.loc[frame.practice_id.isin(reference_ids)].copy()


def align_to_reference(
    reference: pd.DataFrame, frame: pd.DataFrame
) -> pd.DataFrame:
    """Align a practice-level output to the complete reference universe."""
    return reference[["practice_id"]].merge(
        frame,
        on="practice_id",
        how="left",
        validate="one_to_one",
    )


# -----------------------------------------------------------------------------
# Summary construction
# -----------------------------------------------------------------------------
def availability_row(
    module: str,
    measure: str,
    cohort: str,
    eligible: pd.Series,
    available: pd.Series,
    source_table: str,
    source_field: str,
) -> dict:
    """Build one availability-summary row from aligned boolean masks."""
    eligible = eligible.astype("boolean").fillna(False).astype(bool)
    available = available.astype("boolean").fillna(False).astype(bool) & eligible
    n_eligible = int(eligible.sum())
    n_available = int(available.sum())

    return {
        "module_order": MODULE_ORDER[module],
        "module": module,
        "measure": measure,
        "cohort": cohort,
        "n_eligible": n_eligible,
        "n_available": n_available,
        "n_unavailable": n_eligible - n_available,
        "pct_available": n_available / n_eligible if n_eligible else pd.NA,
        "source_table": source_table,
        "source_field": source_field,
    }


def build_pipeline_availability(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Summarize availability across modules and for all ESI outputs."""
    reference = frames["table_01_reference"]
    reference_ids = pd.Index(reference.practice_id)
    rows = []

    all_reference = pd.Series(True, index=reference.index)
    rows.append(
        availability_row(
            "Reference",
            "Practices in reference table",
            "All reference practices",
            all_reference,
            all_reference,
            "table_01_reference",
            "practice_id",
        )
    )

    agronomic = align_to_reference(
        reference,
        restrict_to_reference(
            frames["table_03_agronomic_features"], reference_ids
        ),
    )
    maize_rows = agronomic.crops.eq("MAIZE")
    maize_count = int(maize_rows.sum())

    for product, status_col in [
        ("grain", "unit_support_status_grain"),
        ("forage", "unit_support_status_forage"),
    ]:
        eligible = agronomic[status_col].isin(["SUPPORTED", "UNSUPPORTED"])
        rows.append(
            availability_row(
                "Agronomic",
                f"Supported maize {product} yield unit",
                f"Practices reporting maize {product}",
                eligible,
                agronomic[status_col].eq("SUPPORTED"),
                "table_03_agronomic_features",
                status_col,
            )
        )

    water = restrict_to_reference(frames["table_05_b_water_features"], reference_ids)
    water_rows = pd.Series(True, index=water.index)
    rows.append(
        availability_row(
            "Water",
            "Hybrid water-use estimate",
            "Practices represented in the water module",
            water_rows,
            water.water_cons_hyb_ml_dam3.notna(),
            "table_05_b_water_features",
            "water_cons_hyb_ml_dam3",
        )
    )

    fuel = restrict_to_reference(frames["table_06_emissions_fuel"], reference_ids)
    fuel_rows = pd.Series(True, index=fuel.index)
    rows.append(
        availability_row(
            "Fuel",
            "Fuel-emission estimate",
            "Practices represented in the fuel module",
            fuel_rows,
            fuel.fuel_cons_tco2eq.notna(),
            "table_06_emissions_fuel",
            "fuel_cons_tco2eq",
        )
    )

    inputs = restrict_to_reference(
        frames["table_07_emissions_ag_inputs"], reference_ids
    )
    input_rows = pd.Series(True, index=inputs.index)
    for label, flag_col in [
        (
            "Synthetic-input production emissions complete",
            "flag_missing_emissions_syn_prod",
        ),
        (
            "Synthetic-input transport emissions complete",
            "flag_missing_emissions_syn_tran",
        ),
        (
            "Synthetic-input application emissions complete",
            "flag_missing_emissions_syn_app",
        ),
    ]:
        complete = inputs[flag_col].notna() & ~inputs[flag_col].str.contains(
            SYNTHETIC_INPUT_INCOMPLETE_PATTERN, na=False, regex=True
        )
        rows.append(
            availability_row(
                "Inputs",
                label,
                "Practices represented in the agricultural-input module",
                input_rows,
                complete,
                "table_07_emissions_ag_inputs",
                flag_col,
            )
        )

    rows.append(
        availability_row(
            "Inputs",
            "Organic-input application emissions complete",
            "Practices represented in the agricultural-input module",
            input_rows,
            inputs.flag_missing_emissions_org_app.isin(COMPLETE_ORGANIC_STATUSES),
            "table_07_emissions_ag_inputs",
            "flag_missing_emissions_org_app",
        )
    )
    rows.append(
        availability_row(
            "Inputs",
            "Agricultural-input amounts reported only",
            "Practices represented in the agricultural-input module",
            input_rows,
            inputs.flag_imputed_amount.str.contains(
                r"\bREPORTED ONLY\b", na=False, regex=True
            ),
            "table_07_emissions_ag_inputs",
            "flag_imputed_amount",
        )
    )

    residues = align_to_reference(
        reference,
        restrict_to_reference(
            frames["table_08_emissions_residues"], reference_ids
        ),
    )
    rows.append(
        availability_row(
            "Residues",
            "Crop-residue emissions complete",
            "All reference practices",
            all_reference,
            residues.flag_residue_emissions.isin(COMPLETE_RESIDUE_STATUSES),
            "table_08_emissions_residues",
            "flag_residue_emissions",
        )
    )

    soc = align_to_reference(
        reference,
        restrict_to_reference(frames["table_09_emissions_soc"], reference_ids),
    )
    rows.append(
        availability_row(
            "SOC",
            "SOC-change estimate complete",
            "All reference practices",
            all_reference,
            soc.flag_soc_emissions.eq("AVAILABLE"),
            "table_09_emissions_soc",
            "flag_soc_emissions",
        )
    )

    albedo = align_to_reference(
        reference,
        restrict_to_reference(
            frames["table_10_emissions_albedo"], reference_ids
        ),
    )
    rows.append(
        availability_row(
            "Albedo",
            "Albedo-emission estimate",
            "All reference practices",
            all_reference,
            albedo.albedo_emissions_tco2eq.notna(),
            "table_10_emissions_albedo",
            "albedo_emissions_tco2eq",
        )
    )

    net_ghg = align_to_reference(
        reference,
        restrict_to_reference(
            frames["table_11_emissions_net_ghg"], reference_ids
        ),
    )
    for label, value_col in [
        ("Net GHG emissions excluding fuel", "net_ghg_emissions_tco2eq"),
        ("Net GHG emissions including fuel", "net_ghg_emissions_wfuel_tco2eq"),
    ]:
        rows.append(
            availability_row(
                "Net GHG",
                label,
                "All reference practices",
                all_reference,
                net_ghg[value_col].notna(),
                "table_11_emissions_net_ghg",
                value_col,
            )
        )

    esi_scores = align_to_reference(
        reference,
        restrict_to_reference(frames["table_12_esi_scores"], reference_ids),
    )
    esi_rows = []
    for score_col in ESI_SCORE_COLUMNS:
        row = availability_row(
            "ESI scores",
            ESI_SCORE_LABELS[score_col],
            "Maize-only practices",
            maize_rows,
            esi_scores[score_col].notna(),
            "table_12_esi_scores",
            score_col,
        )
        rows.append(row)
        esi_rows.append(row)

    availability = pd.DataFrame(rows).sort_values(
        ["module_order", "measure"], ignore_index=True
    ).drop(columns="module_order")
    esi_availability = pd.DataFrame(esi_rows).drop(columns="module_order")
    return availability, esi_availability, maize_count


def normalized_status(series: pd.Series) -> pd.Series:
    """Normalize missing or blank statuses for transparent counting."""
    status = series.astype("string").str.strip()
    return status.mask(status.isna() | status.eq(""), "MISSING STATUS")


def build_flag_summaries(
    frames: dict[str, pd.DataFrame], reference_ids: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count exact statuses and non-exclusive reasons for every quality flag."""
    status_rows = []
    reason_rows = []

    for module, (table, flags) in FLAG_COLUMNS.items():
        frame = restrict_to_reference(frames[table], reference_ids)
        denominator = frame.practice_id.nunique()

        for flag in flags:
            status = normalized_status(frame[flag])
            counts = status.value_counts(dropna=False)
            for value, count in counts.items():
                status_rows.append(
                    {
                        "module_order": MODULE_ORDER[module],
                        "module": module,
                        "flag": flag,
                        "status": value,
                        "n_records": int(count),
                        "n_flag_records": denominator,
                        "pct_records": count / denominator if denominator else pd.NA,
                        "source_table": table,
                    }
                )

            reason_frame = pd.DataFrame(
                {
                    "practice_id": frame.practice_id,
                    "reason": status.str.split(r"\s+\|\s+"),
                }
            ).explode("reason")
            reason_counts = reason_frame.groupby("reason").practice_id.nunique()
            for reason, count in reason_counts.items():
                reason_rows.append(
                    {
                        "module_order": MODULE_ORDER[module],
                        "module": module,
                        "flag": flag,
                        "reason": reason,
                        "n_records": int(count),
                        "n_flag_records": denominator,
                        "pct_records": count / denominator if denominator else pd.NA,
                        "counts_are_nonexclusive": True,
                        "source_table": table,
                    }
                )

    status_counts = pd.DataFrame(status_rows).sort_values(
        ["module_order", "flag", "n_records"],
        ascending=[True, True, False],
        ignore_index=True,
    ).drop(columns="module_order")
    reason_counts = pd.DataFrame(reason_rows).sort_values(
        ["module_order", "flag", "n_records"],
        ascending=[True, True, False],
        ignore_index=True,
    ).drop(columns="module_order")
    return status_counts, reason_counts


def git_metadata() -> tuple[str, bool | str]:
    """Return the current Git revision and whether the worktree is dirty."""
    scripts_dir = Path(__file__).resolve().parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=scripts_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=scripts_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return revision, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE", "UNAVAILABLE"


def build_run_info(
    database_path: Path,
    reference_count: int,
    maize_count: int,
) -> pd.DataFrame:
    """Create run metadata for interpreting and reproducing the summary."""
    revision, worktree_dirty = git_metadata()
    values = [
        ("generated_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("git_commit", revision),
        ("git_worktree_dirty", worktree_dirty),
        ("python_version", platform.python_version()),
        ("data_root", str(PATH_DATA.resolve())),
        ("sqlite_database", str(database_path.resolve())),
        ("esi_coefficient_table", PATH_ESI_COEFFICIENTS.name),
        ("reference_practices", reference_count),
        ("maize_eligible_practices", maize_count),
        ("record_unit", "Unique practice_id"),
        (
            "cohort_note",
            "Availability percentages use the cohort named in each row.",
        ),
        (
            "reason_count_note",
            "Reason counts are non-exclusive and may exceed unavailable records.",
        ),
    ]
    return pd.DataFrame(values, columns=["property", "value"])


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
def format_workbook(writer: pd.ExcelWriter, sheets: dict[str, pd.DataFrame]) -> None:
    """Apply restrained formatting for readable tabular outputs."""
    header_fill = PatternFill("solid", fgColor="D9EAD3")
    for sheet_name, frame in sheets.items():
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill

        for column_index, column_name in enumerate(frame.columns, 1):
            values = [str(column_name), *frame[column_name].fillna("").astype(str)]
            width = min(max(len(value) for value in values) + 2, 60)
            column_letter = worksheet.cell(1, column_index).column_letter
            worksheet.column_dimensions[column_letter].width = width

            if column_name.startswith("pct_"):
                for row_index in range(2, len(frame) + 2):
                    worksheet.cell(row_index, column_index).number_format = "0.0%"


def export_summaries(
    sheets: dict[str, pd.DataFrame], output_dir: Path
) -> tuple[Path, list[Path]]:
    """Write one Excel workbook and one CSV file per summary sheet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "data_availability_summary.xlsx"
    temporary_workbook = output_dir / "data_availability_summary.tmp.xlsx"

    with pd.ExcelWriter(temporary_workbook, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        format_workbook(writer, sheets)
    temporary_workbook.replace(workbook_path)

    csv_paths = []
    for sheet_name, frame in sheets.items():
        csv_path = output_dir / f"{sheet_name}.csv"
        temporary_csv = output_dir / f".{sheet_name}.tmp.csv"
        frame.to_csv(temporary_csv, index=False)
        temporary_csv.replace(csv_path)
        csv_paths.append(csv_path)

    return workbook_path, csv_paths


def generate_data_availability_summary(
    database_path: Path = PATH_SQL_DB,
    output_dir: Path = PATH_PROCESSED_SUMMARIES,
) -> dict[str, pd.DataFrame]:
    """Build and export all data-availability summary tables."""
    frames = read_pipeline_tables(database_path)
    reference = frames["table_01_reference"]
    reference_ids = pd.Index(reference.practice_id)

    availability, esi_availability, maize_count = build_pipeline_availability(frames)
    status_counts, reason_counts = build_flag_summaries(frames, reference_ids)
    net_ghg_reasons = reason_counts.loc[
        reason_counts.module.eq("Net GHG") & reason_counts.reason.ne("AVAILABLE")
    ].copy()
    run_info = build_run_info(database_path, len(reference), maize_count)

    sheets = {
        "run_info": run_info,
        "pipeline_availability": availability,
        "flag_status_counts": status_counts,
        "flag_reason_counts": reason_counts,
        "net_ghg_reasons": net_ghg_reasons,
        "esi_availability": esi_availability,
    }
    workbook_path, csv_paths = export_summaries(sheets, output_dir)

    print(f"Summary workbook: {workbook_path}")
    print(f"CSV summary tables: {len(csv_paths)}")
    return sheets


if __name__ == "__main__":
    generate_data_availability_summary()
