import sqlite3

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256


class DocDB(object):
    """Sqlite document storage. By default, uses Wikipedia dump from 2023/04/01."""
    def __init__(self, db_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        if len(cursor.fetchall()) == 0:
            assert db_path is not None, f"{self.db_path} is empty"


