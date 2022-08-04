-- List the top three cities with the most number of completed orders

CREATE TABLE trades(order_id integer, user_id integer, price float, 
quantity integer, status char(9), time date, PRIMARY KEY (order_id, user_id));
INSERT into trades(order_id, user_id, price, quantity, status, time) values(10, 
1, 1.3, 1, "completed", "2022-03-03");
INSERT into trades(order_id, user_id, price, quantity, status, time) values(11, 
1, 1.3, 1, "completed", "2021-03-03");
INSERT into trades(order_id, user_id, price, quantity, status, time) values(12, 
1, 1.3, 1, "cancelled", "2022-03-03");
INSERT into trades(order_id, user_id, price, quantity, status, time) values(13, 
1, 1.3, 1, "cancelled", "2022-03-04");
INSERT into trades(order_id, user_id, price, quantity, status, time) values(14, 
2, 1.3, 1, "completed", "2022-03-03");
INSERT into trades(order_id, user_id, price, quantity, status, time) values(15, 
2, 1.3, 1, "cancelled", "2022-03-03");
INSERT into trades(order_id, user_id, price, quantity, status, time) values(16, 
2, 1.3, 1, "cancelled", "2022-03-03");
INSERT into trades(order_id, user_id, price, quantity, status, time) values(17, 
3, 1.3, 1, "completed", "2022-03-03");
INSERT into trades(order_id, user_id, price, quantity, status, time) values(18, 
3, 1.3, 1, "cancelled", "2022-03-03");
INSERT into trades(order_id, user_id, price, quantity, status, time) values(19, 
4, 1.3, 1, "cancelled", "2022-03-03");

CREATE TABLE users(user_id integer, city char(20), email char(20), 
singup_date date, PRIMARY KEY (user_id, city));
INSERT into users(user_id, city, email, singup_date) values(1, "Boston", 
"aa@gmail.com", "2018-01-01");
INSERT into users(user_id, city, email, singup_date) values(2, "Mar del Plata", 
"bb@gmail.com", "2018-01-01");
INSERT into users(user_id, city, email, singup_date) values(3, "Necochea", 
"cc@gmail.com", "2018-01-01");
INSERT into users(user_id, city, email, singup_date) values(4, "Madrid", 
"aa@gmail.com", "2018-01-01");


SELECT
    tab.city
FROM (
    SELECT
        users.city AS city,
        COUNT(users.city) AS n_completed
    FROM
        users LEFT JOIN trades on users.user_id = trades.user_id
    WHERE
        trades.status = 'completed'
    GROUP BY
        users.city
    ) AS tab
ORDER BY 
    n_completed DESC
LIMIT 3;

