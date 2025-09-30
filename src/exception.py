import sys
import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error(error_message, error_detail)

    def get_detailed_error(self, error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            line_number = exc_tb.tb_lineno
            file_name = exc_tb.tb_frame.f_code.co_filename
            return f"Error occurred in script: [{file_name}] at line: [{line_number}] -> {error_message}"
        else:
            return error_message

    def __str__(self):
        return self.error_message
