"""
Q3(c) - Automated Data Quality Tests (pytest)
===============================================
This test suite validates the quality of the Labour Force Survey dataset
using pytest. It is designed to run in a CI/CD pipeline (GitHub Actions)
to ensure data integrity is maintained across every code change.

Tests:
  1. test_no_missing_values       - No null values in any column
  2. test_no_duplicates           - No duplicate rows
  3. test_unemployment_rate_range - u_rate within [0, 100]
  4. test_data_has_minimum_records - At least 100 records
  5. test_date_column_parseable   - 'date' column parseable as datetime

Usage:
  pytest tests/test_data_quality.py -v
"""

import os
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Fixture: Load the dataset once and share across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df():
    """Load the Labour Force Survey dataset for testing.

    Uses the raw data file (data/raw/lfs_month.csv) as the primary source
    since it is always present in the repository. Falls back to the cleaned
    version if available.
    """
    # Resolve paths relative to the project root (one level up from tests/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, "data", "raw", "lfs_month.csv")
    cleaned_path = os.path.join(project_root, "data", "cleaned", "lfs_month_cleaned.csv")

    if os.path.exists(cleaned_path):
        data_path = cleaned_path
    elif os.path.exists(raw_path):
        data_path = raw_path
    else:
        pytest.skip("Dataset not found — skipping data quality tests.")

    return pd.read_csv(data_path)


# ---------------------------------------------------------------------------
# Test 1: No missing values
# ---------------------------------------------------------------------------

def test_no_missing_values(df):
    """Assert that no column contains null / NaN values."""
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    assert cols_with_missing.empty, (
        f"Columns with missing values:\n{cols_with_missing.to_string()}"
    )


# ---------------------------------------------------------------------------
# Test 2: No duplicate rows
# ---------------------------------------------------------------------------

def test_no_duplicates(df):
    """Assert that there are no duplicate rows in the dataset."""
    dup_count = df.duplicated().sum()
    assert dup_count == 0, f"Found {dup_count} duplicate row(s) in the dataset."


# ---------------------------------------------------------------------------
# Test 3: Unemployment rate within valid range
# ---------------------------------------------------------------------------

def test_unemployment_rate_range(df):
    """Assert that u_rate values are between 0 and 100 (inclusive)."""
    assert "u_rate" in df.columns, "'u_rate' column not found in dataset."
    assert df["u_rate"].min() >= 0, (
        f"u_rate has values below 0 (min = {df['u_rate'].min()})."
    )
    assert df["u_rate"].max() <= 100, (
        f"u_rate has values above 100 (max = {df['u_rate'].max()})."
    )


# ---------------------------------------------------------------------------
# Test 4: Minimum number of records
# ---------------------------------------------------------------------------

def test_data_has_minimum_records(df):
    """Assert that the dataset contains at least 100 records."""
    assert len(df) >= 100, (
        f"Dataset has only {len(df)} records; expected at least 100."
    )


# ---------------------------------------------------------------------------
# Test 5: Date column is parseable
# ---------------------------------------------------------------------------

def test_date_column_parseable(df):
    """Assert that the 'date' column can be parsed as datetime."""
    assert "date" in df.columns, "'date' column not found in dataset."
    try:
        pd.to_datetime(df["date"])
    except Exception as e:
        pytest.fail(f"'date' column could not be parsed as datetime: {e}")
