The login credentials are required to make your own database with users and passwords.

For example, create a new MySQL/MariaDB user, and whilst logged into it -
```
MariaDB [(none)]> create database stocks;
Query OK, 1 row affected (0.023 sec)

MariaDB [(none)]> use stocks;
Database changed
MariaDB [stocks]> create table test(email varchar(255), password varchar(255));
Query OK, 0 rows affected (0.153 sec)

MariaDB [stocks]> insert into test values ("rehan", "hello1234") 
Query OK, 1 row affected (0.029 sec)
```

Set the environmental variables MYSQL_HOST to the hostname, MYSQL_USER and MYSQL_PASS to username and password of the new user.
Run using `python Stock-Predictor/main.py`
