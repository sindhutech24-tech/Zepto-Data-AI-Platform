# Data Pipeline Module

## Objective

This module implements a complete catalog data pipeline:

**scrape → clean → convert → normalize → SQLite → SQL → pandas**

The data source is [Books to Scrape](https://books.toscrape.com/), a public scraping-practice website.

The scraper processes the first **5 catalogue pages**, with 20 books per page, giving approximately 100 raw book records. Each book detail page is visited to obtain its category.

## Required fixed currency conversion

The project-defined baseline is:

**1 GBP = 105.50 INR**

This is a fixed artificial project constant. No live currency API is required or used.

`price_inr = price_gbp * 105.50`

## Files

- `scrape_pipeline.py` — scraping, cleaning, conversion and SQLite creation.
- `queries.py` — SQL queries and pandas comparison.
- `books_cleaned.csv` — generated cleaned dataset.
- `books.db` — generated SQLite database.
- `query_outputs.txt` — generated SQL and pandas outputs.
- `requirements.txt` — Python dependencies.

## Installation

From the `data_pipeline` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

First generate the dataset and database:

```bash
python scrape_pipeline.py
```

Then execute the SQL and pandas analysis:

```bash
python queries.py
```

The scripts run end-to-end without manual copy/pasting.

## Cleaning decisions

### Price

The `£` symbol is removed and the value is converted to `float`.

Example:

```text
£51.77 → 51.77
```

### Rating

The site's textual rating is converted using:

```text
One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
```

### Availability

If the availability text contains `In stock`, `in_stock` is `True`; otherwise it is `False`.

### Failed parsing

Numeric parsing errors are converted to missing values using `errors="coerce"` and then filled with the column median. This prevents a single malformed numeric value from crashing the pipeline.

Rows missing required `title` or `category` values are dropped because they cannot be reliably represented in the required relational schema.

## Database schema

### categories

```text
category_id INTEGER PRIMARY KEY
category_name TEXT UNIQUE
```

### books

```text
book_id INTEGER PRIMARY KEY
title TEXT
price_gbp REAL
price_inr REAL
rating INTEGER
in_stock INTEGER
category_id INTEGER FOREIGN KEY → categories.category_id
```

This normalizes category names into a separate table and connects books to categories through a foreign key.

## SQL requirements covered

`queries.py` demonstrates:

1. `SELECT` + `WHERE`
2. `ORDER BY`
3. `LIMIT`
4. `DISTINCT`
5. `BETWEEN`
6. `IN`
7. `JOIN`

The executed outputs are saved to `query_outputs.txt`.

## pandas requirements

Two SQL results are read using `pd.read_sql()`.

The JOIN is also reproduced without SQL by loading the two tables into pandas and using:

```python
books_df.merge(categories_df, on="category_id", how="inner")
```

The script compares the SQL JOIN result and pandas JOIN result and prints whether they are equivalent.

## Expected acceptance checks

- At least 60 books
- At least 3 categories
- Correct numeric `price_gbp`
- Integer `rating` from 1–5
- Boolean `in_stock`
- Correct fixed-rate `price_inr`
- Two-table normalized SQLite schema
- Five or more SQL queries
- JOIN
- `pd.read_sql`
- `pd.merge`
- Saved outputs
