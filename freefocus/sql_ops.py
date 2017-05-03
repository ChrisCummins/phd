"""
SQL schema for FreeFocus.
"""
import freefocus
import sqlalchemy as sql

from freefocus import sql

class System:
    @staticmethod
    def helo():
        return {
            "spec": freefocus.__freefocus_spec__
        }
