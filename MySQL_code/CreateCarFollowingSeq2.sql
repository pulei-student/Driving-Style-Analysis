Drop table if exists CarFollowingSeq2;
SET @num := 0;    
SET @Device := 0;
SET @Trip := 0;
SET @ObstacleID := 0;

CREATE TABLE CarFollowingSeq2 AS SELECT 
	@num := if(@Device = w.Device and @Trip = w.Trip and @ObstacleID = c.ObstacleId, @num, @num + 1)as Num,
	@Device :=`w`.`Device` as Device,
    @Trip := `w`.`Trip` as Trip,
    `w`.`Time`,
    `c`.`TargetId`,
    @ObstacleID := `c`.`ObstacleId` as ObstacleID ,
    `c`.`Range`,
    `c`.`Rangerate`,
    `c`.`Transversal`,
    `c`.`TargetType`,
    `c`.`Status`,
    `c`.`CIPV`,
    `w`.`GpsValidWsu`,
    `w`.`GpsTimeWsu`,
    `w`.`LatitudeWsu`,
    `w`.`LongitudeWsu`,
    `w`.`AltitudeWsu`,
    `w`.`GpsHeadingWsu`,
    `w`.`GpsSpeedWsu`,
    `w`.`HdopWsu`,
    `w`.`PdopWsu`,
    `w`.`FixQualityWsu`,
    `w`.`GpsCoastingWsu`,
    `w`.`ValidCanWsu`,
    `w`.`YawRateWsu`,
    `w`.`SpeedWsu`,
    `w`.`TurnSngRWsu`,
    `w`.`TurnSngLWsu`,
    `w`.`BrakeAbsTcsWsu`,
    `w`.`AxWsu`,
    `w`.`PrndlWsu`,
    `w`.`VsaActiveWsu`,
    `w`.`HeadlampWsu`,
    `w`.`WiperWsu`,
    `w`.`ThrottleWsu`,
    `w`.`SteerWsu`
FROM    `processing`.`CarFollowingSeq` AS c
JOIN `sponeday`.`DataWsu` AS w 
	ON c.﻿Device = w.Device AND c.Trip = w.Trip AND c.Time = w.Time
ORDER BY c.﻿Device,c.Trip,`c`.`ObstacleId`,c.Time;

