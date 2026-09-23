"""Scrape books from books.toscrape.com using requests + BeautifulSoup."""
from __future__ import annotations
from typing import Dict, List
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATEGORY_URLS = {
    "Travel": "catalogue/category/books/travel_2/index.html",
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "catalogue/category/books/historical-fiction_4/index.html",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36"}
TIMEOUT = 20


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_listing_page(soup: BeautifulSoup, category: str, page_url: str) -> List[Dict]:
    records: List[Dict] = []
    for article in soup.select("article.product_pod"):
        title_tag = article.select_one("h3 a")
        price_tag = article.select_one("p.price_color")
        rating_tag = article.select_one("p.star-rating")
        availability_tag = article.select_one("p.instock.availability")

        title = ""
        if title_tag:
            title = title_tag.get("title") or title_tag.get_text(" ", strip=True)

        price = price_tag.get_text(" ", strip=True) if price_tag else ""
        availability = availability_tag.get_text(" ", strip=True) if availability_tag else ""

        star_rating = ""
        if rating_tag:
            rating_words = {"One", "Two", "Three", "Four", "Five"}
            star_rating = next((x for x in rating_tag.get("class", []) if x in rating_words), "")

        records.append({
            "title": title,
            "price": price,
            "star_rating": star_rating,
            "availability": availability,
            "category": category,
        })
    return records


def scrape_category(category: str, relative_url: str) -> List[Dict]:
    current_url = urljoin(BASE_URL, relative_url)
    results: List[Dict] = []
    visited = set()

    while current_url and current_url not in visited:
        visited.add(current_url)
        soup = fetch_soup(current_url)
        results.extend(parse_listing_page(soup, category, current_url))

        next_link = soup.select_one("li.next a")
        current_url = urljoin(current_url, next_link.get("href")) if next_link and next_link.get("href") else None

    return results


def scrape_all() -> List[Dict]:
    all_records: List[Dict] = []
    for category, url in CATEGORY_URLS.items():
        records = scrape_category(category, url)
        print(f"{category}: {len(records)} books scraped")
        all_records.extend(records)
    return all_records
