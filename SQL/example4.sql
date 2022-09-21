-- Compute the time spent for each activity as a percentage of the total time spent

CREATE TABLE activity_types(activity_id integer, user_id integer, types char(4), 
time_spent float, activity_date datetime,
primary key(activity_id, user_id));

INSERT into activity_types(activity_id, user_id, types, time_spent, 
activity_date) values (1, 1, 'send', 3, '2022-09-15');
INSERT into activity_types(activity_id, user_id, types, time_spent, 
activity_date) values (2, 1, 'open', 5, '2022-09-15');
INSERT into activity_types(activity_id, user_id, types, time_spent, 
activity_date) values (3, 1, 'open', 2, '2022-09-15');
INSERT into activity_types(activity_id, user_id, types, time_spent, 
activity_date) values (1, 2, 'open', 3, '2022-09-15');
INSERT into activity_types(activity_id, user_id, types, time_spent, 
activity_date) values (2, 2, 'send', 5, '2022-09-15');
INSERT into activity_types(activity_id, user_id, types, time_spent, 
activity_date) values (3, 2, 'send', 2, '2022-09-15');
INSERT into activity_types(activity_id, user_id, types, time_spent, 
activity_date) values (1, 3, 'open', 8, '2022-09-15');
INSERT into activity_types(activity_id, user_id, types, time_spent, 
activity_date) values (2, 3, 'send', 2, '2022-09-15');

CREATE TABLE user_country(user_id integer, country char(10), PRIMARY KEY(user_id));
INSERT into user_country(user_id, country) values (1, 'USA');
INSERT into user_country(user_id, country) values (2, 'USA');
INSERT into user_country(user_id, country) values (3, 'Argentina');
INSERT into user_country(user_id, country) values (4, 'Italy');

-- repeating calculations

SELECT
    country,
    SUM(IF(types = 'send', time_spent, 0)) as t_send,
    SUM(IF(types = 'open', time_spent, 0)) as t_open,
    SUM(time_spent) as total,
    SUM(IF(types = 'send', time_spent, 0)) / SUM(time_spent) as percentage_send,
    SUM(IF(types = 'open', time_spent, 0)) / SUM(time_spent) as percentage_open
FROM
  activity_types join user_country on activity_types.user_id = user_country.user_id
GROUP by
  country; 

-- reusing calculations without subqueries

SELECT
  country,
  @T1 := SUM(IF(types = 'send', time_spent, 0)) as t_send,
  @T2 := SUM(IF(types = 'open', time_spent, 0)) as t_open,
  @total := (SELECT(@T1 + @T2)) as total,
  (SELECT(@T1 / @total)) as percentage_send,
  (SELECT(@T2 / @total)) as percentage_open
FROM
  activity_types join user_country on activity_types.user_id = user_country.user_id
GROUP by
  country
