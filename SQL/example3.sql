-- List the users who spent more than $50 in their first transaction

CREATE TABLE user_transactions(transaction_id integer, product_id integer, 
user_id integer, spend float, transaction_date date, PRIMARY KEY (transaction_id, product_id));
INSERT into user_transactions(transaction_id, product_id, user_id, spend, 
transaction_date) values(1 ,1 ,1, 45., '2022-03-08');
INSERT into user_transactions(transaction_id, product_id, user_id, spend, 
transaction_date) values(2 ,1 ,1, 51., '2019-03-08');
INSERT into user_transactions(transaction_id, product_id, user_id, spend, 
transaction_date) values(3 ,1 ,1, 30., '2018-03-07');
INSERT into user_transactions(transaction_id, product_id, user_id, spend, 
transaction_date) values(4 ,1 ,2, 10, '2022-03-08');
INSERT into user_transactions(transaction_id, product_id, user_id, spend, 
transaction_date) values(5 ,1 ,3, 51., '2020-03-08');
INSERT into user_transactions(transaction_id, product_id, user_id, spend, 
transaction_date) values(6 ,1 ,3, 45., '2021-03-08');

  
WITH trans_dates AS (
  SELECT
    user_id AS user_id,
    spend AS spend,
    RANK() over (partition BY user_id
                ORDER BY transaction_date) as ordered
  FROM
    user_transactions
)

SELECT 
  user_id
FROM 
  trans_dates
WHERE
  ordered = 1 and spend >= 50;

