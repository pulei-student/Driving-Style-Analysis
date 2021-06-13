select * from final_events
into outfile 'G:/sql_8_database/Uploads/From_SQL.csv'
fields terminated by ','
optionally enclosed by '"'
escaped by '"'
lines terminated by "\r\n";
