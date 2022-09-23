-- Compute for each month, the number of active users that were active the previous month
-- (count monthly active users, MAU)

CREATE TABLE actions(user_id integer, activity char(10), timestamp datetime);
INSERT actions(user_id, activity, timestamp) values (1, 'sing-in', '2021-01-01');
INSERT actions(user_id, activity, timestamp) values (1, 'sing-in', '2021-02-01');
INSERT actions(user_id, activity, timestamp) values (1, 'comment', '2021-02-01');
INSERT actions(user_id, activity, timestamp) values (1, 'sing-in', '2021-03-01');
INSERT actions(user_id, activity, timestamp) values (2, 'sing-in', '2021-01-01');
INSERT actions(user_id, activity, timestamp) values (2, 'sing-in', '2021-02-01');
INSERT actions(user_id, activity, timestamp) values (3, 'sing-in', '2021-01-01');
INSERT actions(user_id, activity, timestamp) values (4, 'sing-in', '2021-03-01');
INSERT actions(user_id, activity, timestamp) values (4, 'sing-in', '2021-01-01');
INSERT actions(user_id, activity, timestamp) values (5, 'sing-in', '2021-05-01');
INSERT actions(user_id, activity, timestamp) values (6, 'sing-in', '2021-04-01');
INSERT actions(user_id, activity, timestamp) values (6, 'sing-in', '2021-05-01');

select
  extract(month from curr_month.timestamp) as month,
  count(distinct curr_month.user_id) as mau
from
  actions curr_month
where
  exists (
        select
          *
        from
          actions last_month
        where
          extract(month from last_month.timestamp) + 1 =
          extract(month from curr_month.timestamp)
          and
          last_month.user_id = curr_month.user_id )
group by
  month
order by
  month asc
