"""End-to-end runner. Run: python -m data_pipeline.run_pipeline"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.scrape_books import scrape_all
from data_pipeline.clean_data import clean_books
from data_pipeline.database import create_database
from data_pipeline.sql_queries import run_and_save_queries
from data_pipeline.validate_pipeline import validate_join_with_pandas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_pipeline" / "data"
DB = ROOT / "data_pipeline" / "database"
OUT = ROOT / "data_pipeline" / "outputs"
RAW = DATA / "raw_books.csv"
CLEAN = DATA / "cleaned_books.csv"
DB_PATH = DB / "books.db"
SQL_OUT = OUT / "sql_outputs.txt"
PANDAS_OUT = OUT / "pandas_merge_validation.txt"


def main() -> None:
    print("=" * 80)
    print("ZEPTO DATA & AI PLATFORM - MODULE 1 DATA PIPELINE")
    print("=" * 80)

    for folder in [DATA, DB, OUT]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Scraping...")
    raw_df = pd.DataFrame(scrape_all())
    if raw_df.empty:
        raise RuntimeError("No rows were scraped.")
    if raw_df["category"].nunique() < 3 or len(raw_df) < 60:
        raise RuntimeError(f"Acceptance requirement not met: {len(raw_df)} rows, {raw_df['category'].nunique()} categories")
    raw_df.to_csv(RAW, index=False)
    print(f"Raw rows: {len(raw_df)} | Categories: {raw_df['category'].nunique()}")

    print("\n[2/6] Cleaning and converting...")
    cleaned_df, stats = clean_books(raw_df)
    if len(cleaned_df) < 60:
        raise RuntimeError("Fewer than 60 rows remain after cleaning.")
    cleaned_df.to_csv(CLEAN, index=False)
    assert cleaned_df["rating"].between(1, 5).all()
    assert cleaned_df["in_stock"].dtype == bool
    expected_price_inr = (cleaned_df["price_gbp"] * 105.50).round(2)

    assert cleaned_df["price_inr"].round(2).equals(expected_price_inr)
    #assert ((cleaned_df["price_inr"] - cleaned_df["price_gbp"] * 105.50).abs() < 0.01).all()
    print(stats)
    print(cleaned_df.dtypes)

    print("\n[3/6] Loading SQLite...")
    create_database(DB_PATH, cleaned_df)
    print(DB_PATH)

    print("\n[4/6] Running SQL...")
    run_and_save_queries(DB_PATH, SQL_OUT)

    print("\n[5/6] Validating SQL JOIN with pandas.merge...")
    validate_join_with_pandas(DB_PATH, PANDAS_OUT)

    print("\n[6/6] Complete")
    print(f"Raw CSV:       {RAW}")
    print(f"Cleaned CSV:   {CLEAN}")
    print(f"SQLite DB:     {DB_PATH}")
    print(f"SQL outputs:   {SQL_OUT}")
    print(f"Pandas output: {PANDAS_OUT}")
    print("\nPIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
