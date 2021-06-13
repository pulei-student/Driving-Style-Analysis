
drop table if exists carfollowingseq3;
CREATE TABLE carfollowing3 as
    select 
 `cf`.`Device`,
 `cf`.`Trip` ,
`cf`.`Time` ,
		`cf`.`Range`,
		`cf`.`Rangerate`,
		`cf`.`TargetType`,
		`cf`.`Status`,
		`cf`.`CIPV`,
		`cf`.`AxWsu`,
		`cf`.`SpeedWsu`,
        `cf`.`ObstacleId`
	FROM `CarFollowingSeq2` as cf
	ORDER BY device,trip,time;