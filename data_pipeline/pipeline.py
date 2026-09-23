from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
GBP_TO_INR = 105.50
OUT_DIR = Path(__file__).resolve().parent
DB_PATH = OUT_DIR / "books.db"
CSV_PATH = OUT_DIR / "books_cleaned.csv"
SQL_PATH = OUT_DIR / "queries.sql"
QUERY_OUTPUT_PATH = OUT_DIR / "query_outputs.txt"
PANDAS_JOIN_PATH = OUT_DIR / "pandas_join_validation.csv"


def fetch_page(page: int) -> BeautifulSoup:
    url = BASE_URL.format(page)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_rating(classes: Iterable[str]) -> int | None:
    names = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    for cls in classes:
        if cls in names:
            return names[cls]
    return None


def scrape_books(pages: int = 5) -> pd.DataFrame:
    rows = []
    for page in range(1, pages + 1):
        soup = fetch_page(page)
        for article in soup.select("article.product_pod"):
            title_tag = article.select_one("h3 a")
            price_tag = article.select_one(".price_color")
            availability_tag = article.select_one(".availability")
            rating_tag = article.select_one("p.star-rating")
            # Book pages in this catalogue expose category only on the detail page.
            # We make one detail request per book so the output includes the required category.
            detail_href = title_tag.get("href", "") if title_tag else ""
            detail_url = requests.compat.urljoin(BASE_URL.format(page), detail_href)
            detail_response = requests.get(detail_url, timeout=30)
            detail_response.raise_for_status()
            detail_soup = BeautifulSoup(detail_response.text, "html.parser")
            breadcrumb = [x.get_text(strip=True) for x in detail_soup.select("ul.breadcrumb li")]
            category = breadcrumb[-2] if len(breadcrumb) >= 2 else "Unknown"
            rows.append(
                {
                    "title": title_tag.get("title", title_tag.get_text(strip=True)) if title_tag else None,
                    "price": price_tag.get_text(strip=True) if price_tag else None,
                    "star_rating": parse_rating(rating_tag.get("class", [])) if rating_tag else None,
                    "availability": availability_tag.get_text(" ", strip=True) if availability_tag else None,
                    "category": category,
                }
            )
    return pd.DataFrame(rows)


def parse_price(value) -> float | None:
    if pd.isna(value):
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    return float(match.group(1)) if match else None


def parse_availability(value) -> bool | None:
    if pd.isna(value):
        return None
    return "in stock" in str(value).lower()


def clean_books(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_gbp"] = out["price"].apply(parse_price)
    out["rating"] = pd.to_numeric(out["star_rating"], errors="coerce")
    out["in_stock"] = out["availability"].apply(parse_availability)

    # Parsing policy: malformed numeric values receive the column median; rows missing
    # title/category are unusable for a relational catalogue and are dropped.
    for col in ["price_gbp", "rating"]:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    if out["in_stock"].isna().any():
        out["in_stock"] = out["in_stock"].fillna(False)
    out = out.dropna(subset=["title", "category"]).copy()
    out["rating"] = out["rating"].round().clip(1, 5).astype(int)
    out["in_stock"] = out["in_stock"].astype(bool)
    out["price_gbp"] = out["price_gbp"].astype(float)
    out["price_inr"] = out["price_gbp"] * GBP_TO_INR
    return out[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def create_database(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0,1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id) REFERENCES categories(category_id)
            );
            """
        )
        categories = sorted(df["category"].dropna().unique().tolist())
        conn.executemany("INSERT INTO categories(category_name) VALUES (?)", [(c,) for c in categories])
        mapping = {row[1]: row[0] for row in conn.execute("SELECT category_id, category_name FROM categories")}
        book_rows = [
            (
                row.title,
                row.price_gbp,
                row.price_inr,
                int(row.rating),
                int(row.in_stock),
                mapping[row.category],
            )
            for row in df.itertuples()
        ]
        conn.executemany(
            """INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            book_rows,
        )
        conn.commit()
    finally:
        conn.close()


def run_queries(db_path: Path = DB_PATH) -> dict[str, list[tuple]]:
    queries = {
        "q1_select_where": "SELECT title, price_gbp FROM books WHERE price_gbp > 20 ORDER BY price_gbp DESC LIMIT 10;",
        "q2_distinct": "SELECT DISTINCT rating FROM books ORDER BY rating;",
        "q3_in": "SELECT title, category_id FROM books WHERE rating IN (4, 5) ORDER BY rating DESC, title LIMIT 10;",
        "q4_between": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 20 ORDER BY price_gbp LIMIT 10;",
        "q5_join": "SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.rating DESC, c.category_name, b.title LIMIT 10;",
        "q6_summary": "SELECT c.category_name, COUNT(*) AS book_count, AVG(b.price_gbp) AS avg_price_gbp FROM books b JOIN categories c ON b.category_id = c.category_id GROUP BY c.category_name ORDER BY book_count DESC;",
    }
    SQL_PATH.write_text("\n".join(f"-- {name}\n{sql}\n" for name, sql in queries.items()), encoding="utf-8")
    outputs = {}
    log_lines = []
    conn = sqlite3.connect(db_path)
    try:
        for name, sql in queries.items():
            cur = conn.execute(sql)
            rows = cur.fetchall()
            outputs[name] = rows
            print(f"\n{name}\n{sql}\nOUTPUT:")
            log_lines.append(f"{name}\n{sql}\nOUTPUT:")
            for row in rows:
                print(row)
                log_lines.append(repr(row))
    finally:
        conn.close()
    QUERY_OUTPUT_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    return outputs


def validate_pandas_equivalence(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        sql_join = (
            "SELECT c.category_name, b.title, b.rating, b.price_inr "
            "FROM books b JOIN categories c ON b.category_id = c.category_id "
            "ORDER BY b.rating DESC, c.category_name, b.title LIMIT 10"
        )
        sql_df = pd.read_sql(sql_join, conn)
        books_df = pd.read_sql("SELECT book_id, title, rating, price_inr, category_id FROM books", conn)
        categories_df = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
    finally:
        conn.close()

    merged = books_df.merge(categories_df, on="category_id", how="inner")
    merged = merged[["category_name", "title", "rating", "price_inr"]].sort_values(
        ["rating", "category_name", "title"], ascending=[False, True, True]
    ).head(10).reset_index(drop=True)
    sql_df = sql_df.reset_index(drop=True)
    print("\nJOIN via pd.read_sql:")
    print(sql_df.to_string(index=False))
    print("\nJOIN via pandas.merge:")
    print(merged.to_string(index=False))
    combined = sql_df.copy()
    combined.columns = [f"sql_{c}" for c in combined.columns]
    merged_out = merged.copy()
    merged_out.columns = [f"pandas_{c}" for c in merged_out.columns]
    pd.concat([combined.reset_index(drop=True), merged_out.reset_index(drop=True)], axis=1).to_csv(PANDAS_JOIN_PATH, index=False)
    assert sql_df.equals(merged), "SQL JOIN and pandas.merge outputs differ"


def main() -> None:
    raw = scrape_books(pages=5)
    print(f"Scraped {len(raw)} rows across {raw['category'].nunique()} categories before cleaning.")
    if len(raw) < 60 or raw["category"].nunique() < 3:
        raise RuntimeError("The required minimum of 60 books across 3 categories was not met.")
    cleaned = clean_books(raw)
    cleaned.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(cleaned)} cleaned books to {CSV_PATH}")
    create_database(cleaned)
    run_queries()
    validate_pandas_equivalence(cleaned)
    print(f"Created normalized SQLite database at {DB_PATH}")


if __name__ == "__main__":
    main()
