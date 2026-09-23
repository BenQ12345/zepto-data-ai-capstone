# Data Pipeline

## Run

```bash
python pipeline.py
```

The pipeline requests the first five pagination pages from `books.toscrape.com`, fetches each book's detail page to obtain its category, cleans the scraped fields, and writes `books_cleaned.csv` and `books.db`.

## Cleaning decisions

- `price` is parsed into `price_gbp` as `float`.
- Text star labels are converted to integer `rating` values 1–5.
- Availability is converted to boolean `in_stock`.
- If numeric parsing fails, the malformed numeric value is replaced by that numeric column's median. This keeps the pipeline resilient to isolated dirty cells without inventing a row-level drop for otherwise usable records.
- Rows missing a title or category are dropped because those are required business dimensions for the relational catalogue.
- `price_inr` uses the exact required fixed baseline: **1 GBP = 105.50 INR**.

## Relational design

`categories(category_id)` is referenced by `books(category_id)`, giving a normalized parent/child relationship. Foreign-key enforcement is enabled in SQLite.

## SQL evidence

`queries.sql` contains six executed queries collectively covering `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `IN`, `BETWEEN`, and `JOIN`. The pipeline prints each query's output. It also reads the SQL JOIN with `pd.read_sql()` and reproduces the same result with `pandas.merge()`; the script asserts equality.
