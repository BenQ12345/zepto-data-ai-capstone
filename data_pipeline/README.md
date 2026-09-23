# Module 1 - Data Pipeline

## Objective

Build a reproducible raw-to-relational pipeline:

`books.toscrape.com -> requests/BeautifulSoup -> clean pandas DataFrame -> GBP/INR conversion -> SQLite -> SQL -> pandas validation`

## Scope

This implementation scrapes:

- Travel
- Mystery
- Historical Fiction

Each category is followed through all its paginated listing pages until no next page remains. The final dataset therefore exceeds the required 60 books.

## Fixed conversion

**1 GBP = 105.50 INR**

This is the required project-defined constant.
