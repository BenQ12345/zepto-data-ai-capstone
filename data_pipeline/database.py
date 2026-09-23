"""SQLite schema, loading and read helpers."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def create_database(db_path: Path, books_df: pd.DataFrame) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)
        categories = books_df[["category"]].drop_duplicates().sort_values("category")
        conn.executemany("INSERT INTO categories(category_name) VALUES (?)", [(x,) for x in categories["category"]])
        category_map = dict(conn.execute("SELECT category_name, category_id FROM categories").fetchall())
        rows = [
            (r.title, float(r.price_gbp), float(r.price_inr), int(r.rating), int(bool(r.in_stock)), category_map[r.category])
            for r in books_df.itertuples(index=False)
        ]
        # Final database safety check
        required_numeric = [
            "price_gbp",
            "price_inr",
            "rating"
        ]

        for column in required_numeric:
            if books_df[column].isna().any():
                raise ValueError(
                    f"Column {column} contains missing values before SQLite insert."
                )
        conn.executemany(
            "INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def execute_query(db_path: Path, query: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        return pd.read_sql(query, conn)
