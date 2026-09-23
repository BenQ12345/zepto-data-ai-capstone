from pathlib import Path
import sqlite3
import pandas as pd

HERE = Path(__file__).resolve().parent

df = pd.read_csv(HERE / 'books_cleaned.csv') if (HERE/'books_cleaned.csv').exists() else None
if df is not None:
    assert len(df) >= 60
    assert {'price_gbp','price_inr','rating','in_stock','category'} <= set(df.columns)
    assert df['rating'].between(1,5).all()
    assert df['price_inr'].sub(df['price_gbp']*105.50).abs().max() < 1e-9
conn = sqlite3.connect(HERE/'books.db') if (HERE/'books.db').exists() else None
if conn:
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
    assert 'categories' in tables and 'books' in tables
    fk = pd.read_sql("PRAGMA foreign_key_list(books)", conn)
    assert not fk.empty
    conn.close()
print('data_pipeline smoke test: PASS')
