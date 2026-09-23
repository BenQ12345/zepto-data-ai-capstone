-- q1_select_where
SELECT title, price_gbp FROM books WHERE price_gbp > 20 ORDER BY price_gbp DESC LIMIT 10;

-- q2_distinct
SELECT DISTINCT rating FROM books ORDER BY rating;

-- q3_in
SELECT title, category_id FROM books WHERE rating IN (4, 5) ORDER BY rating DESC, title LIMIT 10;

-- q4_between
SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 20 ORDER BY price_gbp LIMIT 10;

-- q5_join
SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.rating DESC, c.category_name, b.title LIMIT 10;

-- q6_summary
SELECT c.category_name, COUNT(*) AS book_count, AVG(b.price_gbp) AS avg_price_gbp FROM books b JOIN categories c ON b.category_id = c.category_id GROUP BY c.category_name ORDER BY book_count DESC;
