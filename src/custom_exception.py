import traceback
import sys

class CustomException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        self.trace = self._get_traceback()

    def _get_traceback(self):
        tb = sys.exc_info()[2]
        if tb is not None:
            return ''.join(traceback.format_tb(tb))
        return "No traceback available."

    def __str__(self):
        return f"{self.message}\nTraceback:\n{self.trace}"