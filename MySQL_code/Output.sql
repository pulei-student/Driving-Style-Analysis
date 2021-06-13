select * from final_events
into outfile 'G:/sql_8_database/Uploads/event_with_num.csv'
fields terminated by ','
optionally enclosed by '"'
escaped by '"'
lines terminated by "\r\n";