"""
Cleaning and transformation logic for the Zepto data pipeline.
"""

from __future__ import annotations

import re

import pandas as pd

import numpy as np


# Required project-defined conversion rate
GBP_TO_INR = 105.50


RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def parse_price(value):
    """
    Convert raw price text into a float.

    Handles examples such as:
        £51.77
        Â£51.77
        51.77
        " £51.77 "
    """

    if pd.isna(value):
        return float("nan")

    text = str(value).strip()

    # Extract the first numeric price from the text.
    # This is more robust than only replacing the £ symbol.
    match = re.search(r"\d+(?:\.\d+)?", text)

    if match:
        try:
            return float(match.group())
        except ValueError:
            return float("nan")

    return float("nan")


def parse_rating(value):
    """
    Convert textual star rating into an integer from 1 to 5.
    """

    if pd.isna(value):
        return float("nan")

    text = str(value).strip().lower()

    # Handle values such as:
    # "Three"
    # "three stars"
    # "Three "
    first_word = text.split()[0] if text else ""

    return RATING_MAP.get(first_word, float("nan"))


def parse_stock(value):
    """
    Convert availability text into True/False.
    """

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return text.startswith("in stock")


def clean_books(raw_df: pd.DataFrame):
    """
    Clean and transform the raw scraped DataFrame.

    Returns:
        cleaned_df
        stats
    """

    df = raw_df.copy()

    original_rows = len(df)

    # ---------------------------------------------------------
    # 1. Validate expected raw columns
    # ---------------------------------------------------------

    required_raw_columns = {
        "title",
        "price",
        "star_rating",
        "availability",
        "category",
    }

    missing_columns = required_raw_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing raw columns: {sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 2. Clean text columns
    # ---------------------------------------------------------

    df["title"] = (
        df["title"]
        .astype("string")
        .str.strip()
    )

    df["category"] = (
        df["category"]
        .astype("string")
        .str.strip()
    )

    # Remove rows with missing essential text fields
    before_drop = len(df)

    df = df.dropna(
        subset=["title", "category"]
    )

    df = df[
        (df["title"] != "") &
        (df["category"] != "")
    ]

    dropped_missing_text = (
        before_drop - len(df)
    )

    # ---------------------------------------------------------
    # 3. Parse GBP price
    # ---------------------------------------------------------

    df["price_gbp"] = df["price"].apply(
        parse_price
    )

    invalid_price_count = int(
        df["price_gbp"].isna().sum()
    )

    print(
        f"Invalid price values before imputation: "
        f"{invalid_price_count}"
    )

    # ---------------------------------------------------------
    # 4. Median imputation for invalid numeric prices
    # ---------------------------------------------------------

    valid_prices = df["price_gbp"].dropna()

    if valid_prices.empty:

        print("\nDEBUG: No valid prices were parsed.")

        print("\nRaw price values:")
        print(
            df["price"]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "No valid price values could be parsed "
            "from the scraped data."
        )

    median_price = valid_prices.median()

    df["price_gbp"] = (
        df["price_gbp"]
        .fillna(median_price)
        .astype(float)
    )

    # Final safety check
    if df["price_gbp"].isna().any():

        raise ValueError(
            "price_gbp still contains NaN after "
            "median imputation."
        )

    # ---------------------------------------------------------
    # 5. Parse star rating
    # ---------------------------------------------------------

    df["rating"] = df["star_rating"].apply(
        parse_rating
    )

    invalid_rating_count = int(
        df["rating"].isna().sum()
    )

    valid_ratings = df["rating"].dropna()

    if valid_ratings.empty:

        # Safe fallback for completely unexpected rating text
        rating_median = 3

    else:

        rating_median = int(
            round(valid_ratings.median())
        )

    df["rating"] = (
        df["rating"]
        .fillna(rating_median)
        .astype(int)
    )

    # Ensure rating stays within assignment requirement
    df["rating"] = (
        df["rating"]
        .clip(lower=1, upper=5)
        .astype(int)
    )

    # ---------------------------------------------------------
    # 6. Parse availability
    # ---------------------------------------------------------

    df["in_stock"] = (
        df["availability"]
        .apply(parse_stock)
        .astype(bool)
    )

    # ---------------------------------------------------------
    # 7. Required GBP -> INR conversion
    # ---------------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # ---------------------------------------------------------
    # 8. Final validations
    # ---------------------------------------------------------

    if df["price_gbp"].isna().any():
        raise ValueError(
            "price_gbp contains NaN values."
        )

    if df["price_inr"].isna().any():
        raise ValueError(
            "price_inr contains NaN values."
        )

    if df["rating"].isna().any():
        raise ValueError(
            "rating contains NaN values."
        )

    # ---------------------------------------------------------
    # 9. Keep only assignment-required columns
    # ---------------------------------------------------------

    cleaned_df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ].copy()

    # Make sure the types are exactly what we want
    cleaned_df["price_gbp"] = (
        cleaned_df["price_gbp"]
        .astype(float)
    )

    cleaned_df["price_inr"] = (
        cleaned_df["price_inr"]
        .astype(float)
    )

    cleaned_df["rating"] = (
        cleaned_df["rating"]
        .astype(int)
    )

    cleaned_df["in_stock"] = (
        cleaned_df["in_stock"]
        .astype(bool)
    )

    # ---------------------------------------------------------
    # 10. Final debug information
    # ---------------------------------------------------------

    print("\nCleaning validation:")
    print(
        cleaned_df.isna().sum()
    )

    print("\nSample cleaned prices:")

    print(
        cleaned_df[
            [
                "price_gbp",
                "price_inr"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 11. Statistics
    # ---------------------------------------------------------

    stats = {
        "original_rows": original_rows,
        "dropped_missing_text_rows": dropped_missing_text,
        "invalid_price_values": invalid_price_count,
        "invalid_rating_values": invalid_rating_count,
        "final_rows": len(cleaned_df),
    }

    return cleaned_df, stats