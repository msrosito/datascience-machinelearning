-- Obtain the number of companies that posted duplicated jobs
-- (two jobs at the same company with the same title and the same description)

CREATE TABLE job_listing(job_id integer, company_id integer, title char(10), 
description char(30), postdate date,  PRIMARY KEY (job_id, company_id));
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (1, 1, "DS", "data scientist", '2021-03-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (2, 1, "DS", "data scientist", '2022-03-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (3, 1, "DS", "data scientist", '2020-03-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (4, 1, "ML", "machine learning eng", '2021-03-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (5, 2, "DS", "data scientist", '2021-03-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (6, 2, "DS", "data scientist", '2022-03-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (7, 2, "Director", "Direction", '2021-02-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (8, 3, "SE", "software engenieer", '2022-03-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (9, 3, "RS", "research scientist", '2022-03-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (10, 4, "RSii", "research scientist II", '2022-03-07');
INSERT into job_listing(job_id, company_id, title, description, postdate) 
values (11, 4, "RSii", "research scientist II", '2022-03-07');

SELECT
    COUNT(DISTINCT t.company_id)
FROM (
    SELECT
        job_listing.company_id as company_id,
        job_listing.title,
        job_listing.description,
        COUNT(*) as djob
    FROM
        job_listing
    GROUP BY
        job_listing.company_id,
        job_listing.title,
        job_listing.description
    ) as t
WHERE
    t.djob >= 2;
