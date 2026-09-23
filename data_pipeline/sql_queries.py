"""Required SQL queries plus captured output."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

QUERIES = {
    "Q1_SELECT_WHERE": """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
ORDER BY rating DESC, title;
""",
    "Q2_ORDER_BY": """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC;
""",
    "Q3_LIMIT": """
SELECT title, price_inr
FROM books
ORDER BY price_inr DESC
LIMIT 10;
""",
    "Q4_DISTINCT": """
SELECT DISTINCT rating
FROM books
ORDER BY rating;
""",
    "Q5_BETWEEN": """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
""",
    "Q6_JOIN": """
SELECT b.title, c.category_name, b.rating, b.price_gbp, b.price_inr
FROM books AS b
JOIN categories AS c ON b.category_id = c.category_id
ORDER BY b.rating DESC, c.category_name, b.title;
""",
}


def run_and_save_queries(db_path: Path, output_path: Path) -> dict[str, pd.DataFrame]:
    results = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn, output_path.open("w", encoding="utf-8") as out:
        out.write("ZEPTO DATA PIPELINE - SQL QUERY OUTPUTS\n" + "=" * 80 + "\n\n")
        for name, query in QUERIES.items():
            df = pd.read_sql(query, conn)
            results[name] = df
            out.write(f"{name}\n{'-' * 80}\n")
            out.write(" ".join(query.split()) + "\n\n")
            out.write(df.to_string(index=False) + "\n\n")
            print(f"\n{name}\n{' '.join(query.split())}\n{df.to_string(index=False)}")
    return results
