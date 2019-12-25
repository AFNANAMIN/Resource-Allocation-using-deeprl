import logging
import sys
from logging.handlers import SocketHandler


class LabelFilter(logging.Filter):

    def filter(self, record):
        if not hasattr(record, 'label'):
            record.label = 'None'
        return True


class EnvLogger:

    def __init__(self):
        self._logger = logging.getLogger('env')
        self._logger.setLevel(logging.INFO)
        self._formatter = logging.Formatter('%(levelname)s:%(name)s:%(label)s:%(message)s')
        self._logger.addFilter(LabelFilter())

        # hd = SocketHandler('127.0.0.1', 19996)

    def set_file_handler(self, dir_log, f_name):
        hd = logging.FileHandler(dir_log+'/'+f_name)
        hd.setFormatter(self._formatter)
        self._logger.addHandler(hd)

    def set_stream_handler(self):
        hd = logging.StreamHandler(sys.stdout)
        hd.setFormatter(self._formatter)
        self._logger.addHandler(hd)

    @property
    def logger(self):
        return self._logger