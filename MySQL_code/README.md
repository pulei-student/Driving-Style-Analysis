# The oder of running files.
### create_data
### car_following_one
### car_following_two
### car_following_specific
### create_final
### get_event_num
### output
# What should I mind befor running:
1. Deffieret softwares can recognize the codes differetly. So please check the codes and get the meaning in it. Or just use MySQL workbench.
2. To clarify, my MySQL version is 8.027. Please check if your version was too old to run the code.
3. Because that the size of data is too big, please leave enough space both in your system and data disk. The recommded space is *System  50G* and *Data disk 100G*. Of course, we will not run out of it when processing data. But we may need to sacriface space to speed up processing because of the big size of data.
4. To speed up, my setting is as follows:
set global innodb_buffer_pool_size=1024\*1024\*1024\*4;
set session BULK_INSERT_BUFFER_SIZE=1024\*1024\*1024\*3;
set UNIQUE_CHECKS=0;
set foreign_key_checks=0;
5. Set all time wait out variables to be 0, which means that skip checking them.
