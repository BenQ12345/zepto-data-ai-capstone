<<<<<<< HEAD
# Zepto Data & AI Platform

Single repository for the three capstone modules:

- `/data_pipeline` - scraping, cleaning, GBP/INR conversion, SQLite, SQL and pandas validation.
- `/analytics` - customer-style analytics and modeling.
- `/support_assistant` - grounded GenAI support assistant.

## Module 1: Data Pipeline

### Python
Python 3.10+ is recommended.

This repository uses one consolidated `requirements.txt` at the repository root.

### Install

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD:
```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run Module 1

From the repository root:

```bash
python -m data_pipeline.run_pipeline
```

You can also run:

```bash
python data_pipeline/run_pipeline.py
```

The pipeline scrapes at least three categories, cleans the records, converts GBP to INR using the required fixed rate `1 GBP = 105.50 INR`, loads SQLite, runs the required SQL queries, and validates the JOIN with `pd.merge`.

### Generated outputs

- `data_pipeline/data/raw_books.csv`
- `data_pipeline/data/cleaned_books.csv`
- `data_pipeline/database/books.db`
- `data_pipeline/outputs/sql_outputs.txt`
- `data_pipeline/outputs/pandas_merge_validation.txt`

### Cleaning decisions

- `price_gbp`: remove `£` and convert to float.
- `rating`: map One/Five text values to integers 1-5.
- `in_stock`: parse availability text into a boolean.
- Numeric parsing failures are median-imputed.
- Rows missing essential title/category fields are dropped.
- `price_inr` always uses the project-defined fixed rate `105.50` INR per GBP; no external currency API is required.

## Git workflow requirement

For the repository-wide requirement, create a feature branch, make at least two commits, and merge it into `main`:

```bash
git checkout -b feature/data-pipeline
git add .
git commit -m "Add scraping and cleaning pipeline"
git add .
git commit -m "Add SQLite queries and pandas validation"
git checkout main
git merge --no-ff feature/data-pipeline -m "Merge data pipeline feature"
```
=======
# zepto-data-pipeline
Zepto_Data
>>>>>>> e7dff0ed4a26519d78228e452ee16909ccfad8ed
