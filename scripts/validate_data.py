"""
Q3(a) - Automated Data Quality Validation Script
=================================================
This script performs structural and value-range checks on the Labour Force
Survey dataset. It is executed in the CI/CD pipeline (GitHub Actions) to ensure
no low-quality or corrupted datasets are merged into the main branch.

Checks performed:
  1. Missing Values: No null/NaN values are allowed in any column.
  2. Duplicate Records: No duplicate rows are allowed.
  3. Data Type Validation:
     - 'date' column must be parseable as datetime.
     - Metrics columns must be numeric.
  4. Value Ranges:
     - Unemployment rate ('u_rate'), participation rate ('p_rate'), and
       employment-to-population ratio ('ep_ratio') must lie within [0, 100].
  5. Minimum Records: At least 100 records must exist in the dataset.

Exit Codes:
  0 - All checks passed successfully.
  1 - One or more validation checks failed.
"""

import os
import sys
import pandas as pd
import numpy as np

def run_validation():
    # Resolve paths relative to the project root (one level up from scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, "data", "raw", "lfs_month.csv")
    cleaned_path = os.path.join(project_root, "data", "cleaned", "lfs_month_cleaned.csv")

    if os.path.exists(cleaned_path):
        data_path = cleaned_path
        print(f"[INFO] Using cleaned dataset: {data_path}")
    elif os.path.exists(raw_path):
        data_path = raw_path
        print(f"[INFO] Using raw dataset: {data_path}")
    else:
        print("[ERROR] No dataset found at expected paths!")
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"[ERROR] Failed to read dataset: {e}")
        sys.exit(1)

    validation_failed = False

    # 1. Missing Values Check
    print("-" * 60)
    print("Check 1 — Missing Values")
    print("-" * 60)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"❌ Missing values detected:\n{missing[missing > 0].to_string()}")
        validation_failed = True
    else:
        print("✅ No missing values detected in any column.")

    # 2. Duplicate Records Check
    print("\n" + "-" * 60)
    print("Check 2 — Duplicate Records")
    print("-" * 60)
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"❌ {n_dupes} duplicate record(s) found.")
        validation_failed = True
    else:
        print("✅ No duplicate records found.")

    # 3. Data Type Validation
    print("\n" + "-" * 60)
    print("Check 3 — Data Type Validation")
    print("-" * 60)
    
    # Check date column parseability
    if "date" not in df.columns:
        print("❌ 'date' column is missing from dataset.")
        validation_failed = True
    else:
        try:
            pd.to_datetime(df["date"])
            print("✅ 'date' column is parseable as datetime.")
        except Exception as e:
            print(f"❌ 'date' column could not be parsed as datetime: {e}")
            validation_failed = True

    # Check numeric columns
    numeric_cols = ["lf", "lf_employed", "lf_unemployed", "lf_outside", "p_rate", "ep_ratio", "u_rate"]
    for col in numeric_cols:
        if col not in df.columns:
            print(f"❌ Numeric column '{col}' is missing from dataset.")
            validation_failed = True
        elif not np.issubdtype(df[col].dtype, np.number):
            print(f"❌ Column '{col}' is not numeric (dtype = {df[col].dtype}).")
            validation_failed = True
        else:
            print(f"✅ Column '{col}' is numeric ({df[col].dtype}).")

    # 4. Value Ranges Check
    print("\n" + "-" * 60)
    print("Check 4 — Value Ranges (Percentages / Ratios)")
    print("-" * 60)
    for col in ["u_rate", "p_rate", "ep_ratio"]:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < 0 or max_val > 100:
                print(f"❌ Column '{col}' contains values outside [0, 100] range: min = {min_val}, max = {max_val}.")
                validation_failed = True
            else:
                print(f"✅ Column '{col}' is within [0, 100] range: min = {min_val}, max = {max_val}.")

    # 5. Minimum Records Check
    print("\n" + "-" * 60)
    print("Check 5 — Minimum Records Count")
    print("-" * 60)
    min_records = 100
    if len(df) < min_records:
        print(f"❌ Dataset contains {len(df)} records, which is less than the expected minimum of {min_records}.")
        validation_failed = True
    else:
        print(f"✅ Dataset contains {len(df)} records (minimum required: {min_records}).")

    # Final result
    print("\n" + "=" * 60)
    if validation_failed:
        print("❌ DATA VALIDATION FAILED")
        print("=" * 60)
        sys.exit(1)
    else:
        print("✅ ALL DATA VALIDATION CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    run_validation()
