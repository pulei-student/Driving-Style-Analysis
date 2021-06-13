DROP TABLE IF exists final_events;
SET @NUM := 0;
SET @Device := 0;
SET @Trip := 0;
Set @Time := 0; 
SET @OID := 0;


CREATE TABLE final_events as

    select 
		@NUM := if(@Device = cf.Device and @Trip = cf.Trip and @Time = cf.Time - 10 and @OID = cf.obstacleID, @NUM, @NUM + 1)	as eventNum,
		@Device := `cf`.`Device` as device,
		@Trip := `cf`.`Trip` as trip,
		@Time := `cf`.`Time` as time,
		@OId :=`cf`.`ObstacleId` as ObstacleId,
		`cf`.`Range`,
		`cf`.`Rangerate`,
		`cf`.`TargetType`,
		`cf`.`Status`,
		`cf`.`CIPV`,
        `cf`.`AxWsu`,
        `cf`.`SpeedWsu`
	FROM `final` as cf
	ORDER BY device,trip,obstacleID,time;