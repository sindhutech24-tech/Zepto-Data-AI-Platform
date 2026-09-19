import sqlite3
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com"
CATALOGUE_URL = f"{BASE_URL}/catalogue"
GBP_TO_INR = 105.50
DB_PATH = Path(__file__).parent / "books.db"
CSV_PATH = Path(__file__).parent / "books_cleaned.csv"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(url):
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (data-pipeline-assignment)"}
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def scrape_books(max_pages=5):
    """Scrape the first five catalogue pages: 5 x 20 = 100 books."""
    rows = []

    for page in range(1, max_pages + 1):
        url = f"{CATALOGUE_URL}/page-{page}.html"
        soup = get_soup(url)

        for product in soup.select("article.product_pod"):
            title = product.select_one("h3 a")["title"].strip()
            price = product.select_one(".price_color").get_text(strip=True)
            rating_text = next(
                (cls.title() for cls in product.select_one(".star-rating")["class"]
                 if cls in RATING_MAP),
                None
            )
            availability = product.select_one(".availability").get_text(" ", strip=True)

 
            book_href = product.select_one("h3 a")["href"]
            book_url = requests.compat.urljoin(url, book_href)
            detail_soup = get_soup(book_url)

            breadcrumb = detail_soup.select("ul.breadcrumb li")
            category = breadcrumb[-2].get_text(strip=True) if len(breadcrumb) >= 2 else None

            rows.append({
                "title": title,
                "price": price,
                "star_rating": rating_text,
                "availability": availability,
                "category": category,
            })

        print(f"Scraped page {page}: {len(rows)} total books")

    return pd.DataFrame(rows)


def clean_data(df):
    df = df.copy()

   
    df["price_gbp"] = pd.to_numeric(
        df["price"].astype(str).str.replace("£", "", regex=False).str.strip(),
        errors="coerce"
    )

   
    df["rating"] = df["star_rating"].map(RATING_MAP)

    df["in_stock"] = (
        df["availability"]
        .astype(str)
        .str.contains(r"\bIn stock\b", case=False, regex=True, na=False)
    )

    for col in ["price_gbp", "rating"]:
        if df[col].isna().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    
    df = df.dropna(subset=["title", "category"]).copy()

    
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    columns = [
        "title", "price_gbp", "rating", "in_stock",
        "price_inr", "category"
    ]
    df = df[columns]

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["price_inr"] = df["price_inr"].astype(float)

    return df


def create_database(df):
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    conn.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    conn.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)

    categories = sorted(df["category"].unique())
    conn.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(category,) for category in categories]
    )

    category_map = dict(
        conn.execute("SELECT category_name, category_id FROM categories")
    )

    book_rows = [
        (
            row.title,
            row.price_gbp,
            row.price_inr,
            row.rating,
            int(row.in_stock),
            category_map[row.category],
        )
        for row in df.itertuples(index=False)
    ]

    conn.executemany("""
        INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, book_rows)

    conn.commit()
    conn.close()


def main():
    print("Starting Books to Scrape pipeline...")
    raw_df = scrape_books(max_pages=5)

    if len(raw_df) < 60:
        raise RuntimeError(f"Only {len(raw_df)} rows scraped; at least 60 are required.")

    clean_df = clean_data(raw_df)

    if len(clean_df) < 60:
        raise RuntimeError(f"Only {len(clean_df)} clean rows remain; at least 60 are required.")

    if clean_df["category"].nunique() <                                                                                                                                                                                                                                                                                                                                                                                                       3:
        raise RuntimeError("Fewer than 3 categories were found.")

    clean_df.to_csv(CSV_PATH, index=False)
    create_database(clean_df)

    print("\nPipeline completed successfully.")
    print(f"Books: {len(clean_df)}")
    print(f"Categories: {clean_df['category'].nunique()}")
    print(f"Database: {DB_PATH}")
    print(f"Clean CSV: {CSV_PATH}")
    print(f"Fixed conversion rate: 1 GBP = {GBP_TO_INR:.2f} INR")


if __name__ == "__main__":
    main()
