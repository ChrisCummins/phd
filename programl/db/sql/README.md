# Database

```
$ sudo su - postgres
> CREATE ROLE cec WITH LOGIN ENCRYPTED PASSWORD '1234';
> ALTER USER cec CREATEDB

$ dropdb programl
$ createdb programl
$ psql -d programl < schema.sql
```
