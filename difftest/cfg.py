from labm8 import fs
from typing import Tuple
from configparser import ConfigParser

HOSTNAME = "cc1"

DATABASE = "DeepSmith_1"

PORT = 3306

def get_mysql_creds() -> Tuple[str, str]:
    """ read default MySQL credentials in ~/.my.cnf """
    config = ConfigParser()
    config.read(fs.path("~/.my.cnf"))
    return config['mysql']['user'], config['mysql']['password']
