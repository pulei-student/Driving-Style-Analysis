drop table if exists final;
create table final as
select * from carfollowingseq_new3 as cf3
where `cf3`.`range`>5 and `cf3`.`range`<120 and `cf3`.`SpeedWsu`>5
order by `cf3`.`Device`,`cf3`.`Trip`,`cf3`.`Time`;