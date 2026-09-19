import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "books.db"
OUTPUT_PATH = Path(__file__).parent / "query_outputs.txt"


QUERIES = {
    "Q1_SELECT_WHERE": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC, price_gbp DESC
        LIMIT 10;
    """,

    "Q2_ORDER_BY_LIMIT": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    "Q3_DISTINCT": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,

    "Q4_BETWEEN": """
        SELECT title, price_gbp, price_inr
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,

    "Q5_IN": """
        SELECT title, rating, in_stock
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        LIMIT 15;
    """,

    "Q6_JOIN": """
        SELECT
            b.title,
            c.category_name,
            b.rating,
            b.price_gbp,
            b.price_inr,
            b.in_stock
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC, c.category_name, b.title
        LIMIT 20;
    """
}


def run_sql_queries():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    output_parts = []

    for name, query in QUERIES.items():
        df = pd.read_sql(query, conn)
        output_parts.append(f"\n{'=' * 70}\n{name}\n{'=' * 70}\n")
        output_parts.append(df.to_string(index=False))
        print(f"\n{name}")
        print(df.to_string(index=False))

    # Read two SQL results back into pandas using pd.read_sql.
    q1_df = pd.read_sql(QUERIES["Q1_SELECT_WHERE"], conn)
    q6_sql_df = pd.read_sql(QUERIES["Q6_JOIN"], conn)

    # Reproduce the JOIN entirely in pandas.
    books_df = pd.read_sql("""
        SELECT book_id, title, price_gbp, price_inr, rating, in_stock, category_id
        FROM books
    """, conn)

    categories_df = pd.read_sql("""
        SELECT category_id, category_name
        FROM categories
    """, conn)

    q6_merge_df = (
        books_df
        .merge(categories_df, on="category_id", how="inner")
        .sort_values(["rating", "category_name", "title"],
                     ascending=[False, True, True])
        .head(20)
        [["title", "category_name", "rating", "price_gbp", "price_inr", "in_stock"]]
        .reset_index(drop=True)
    )

    q6_sql_compare = q6_sql_df.reset_index(drop=True)

    equivalent = q6_sql_compare.equals(q6_merge_df)

    output_parts.append(
        f"\n{'=' * 70}\nPANDAS READ_SQL / MERGE COMPARISON\n{'=' * 70}\n"
    )
    output_parts.append("JOIN using pd.read_sql():\n")
    output_parts.append(q6_sql_compare.to_string(index=False))
    output_parts.append("\n\nJOIN reproduced using pd.merge():\n")
    output_parts.append(q6_merge_df.to_string(index=False))
    output_parts.append(f"\n\nEquivalent output: {equivalent}\n")

    OUTPUT_PATH.write_text("\n".join(output_parts), encoding="utf-8")

    print("\nQ1 loaded with pd.read_sql():")
    print(q1_df.to_string(index=False))

    print("\nJOIN using pd.read_sql():")
    print(q6_sql_compare.to_string(index=False))

    print("\nJOIN reproduced with pd.merge():")
    print(q6_merge_df.to_string(index=False))

    print(f"\nEquivalent output: {equivalent}")
    print(f"Saved outputs to: {OUTPUT_PATH}")

    conn.close()


if __name__ == "__main__":
    run_sql_queries()
