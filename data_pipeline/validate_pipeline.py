"""Validate pd.read_sql and pd.merge outputs for the JOIN."""
from __future__ import annotations
from pathlib import Path
import sqlite3
import pandas as pd
from .sql_queries import QUERIES


def validate_join_with_pandas(db_path: Path, output_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        sql_join_df = pd.read_sql(QUERIES["Q6_JOIN"], conn)
        books_df = pd.read_sql("SELECT book_id, title, price_gbp, price_inr, rating, in_stock, category_id FROM books", conn)
        categories_df = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
        high_rated_df = pd.read_sql("SELECT title, rating FROM books WHERE rating >= 4", conn)

    pandas_join_df = pd.merge(books_df, categories_df, on="category_id", how="inner")[[
        "title", "category_name", "rating", "price_gbp", "price_inr"
    ]]

    sort_cols = ["category_name", "title"]
    sql_check = sql_join_df.sort_values(sort_cols).reset_index(drop=True)
    pandas_check = pandas_join_df.sort_values(sort_cols).reset_index(drop=True)
    equivalent = sql_check.equals(pandas_check)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write("ZEPTO DATA PIPELINE - PANDAS VALIDATION\n" + "=" * 80 + "\n\n")
        out.write("pd.read_sql - example result\n" + "-" * 80 + "\n")
        out.write(high_rated_df.to_string(index=False) + "\n\n")
        out.write("JOIN RESULT USING pd.read_sql()\n" + "-" * 80 + "\n")
        out.write(sql_check.to_string(index=False) + "\n\n")
        out.write("JOIN RESULT USING pd.merge()\n" + "-" * 80 + "\n")
        out.write(pandas_check.to_string(index=False) + "\n\n")
        out.write(f"Equivalent outputs: {equivalent}\n")

    print(f"\nJOIN rows via pd.read_sql: {len(sql_check)}")
    print(f"JOIN rows via pd.merge:   {len(pandas_check)}")
    print(f"Equivalent outputs:       {equivalent}")
    if not equivalent:
        raise AssertionError("JOIN result mismatch between SQL and pandas.merge")
