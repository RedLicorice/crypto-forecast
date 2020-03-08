import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from functools import wraps
import inspect
import io
import enlighten
import traceback

def _DummyFn(*args, **kwargs):
    _, _ = args, kwargs
    raise NotImplementedError()

_srcfile = os.path.normcase(_DummyFn.__code__.co_filename)

class WrappedLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def findCaller(self, stack_info=False):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        # get all outer frames starting from the current frame
        outer = inspect.getouterframes(inspect.currentframe())
        # reverse the order, to search from out inward
        outer.reverse()
        rv = "(unknown file)", 0, "(unknown function)"

        this_frame = 0
        # go through all frames
        found = 0
        for i in range(0,len(outer)):
            # stop if we find the current source filename
            if outer[i][1] == _srcfile:
                # found this frame
                this_frame=i
                break
        #the caller frame is the previous frame
        pos = this_frame - 1
        # get the frame (stored in first tuple entry)
        f = outer[pos][0]
        co = f.f_code
        # print("Found: {} {} {}".format(co.co_filename, f.f_lineno, co.co_name))
        sinfo = None
        if stack_info:
            sio = io.StringIO()
            sio.write('Stack (most recent call last):\n')
            traceback.print_stack(f, file=sio)
            sinfo = sio.getvalue()
            if sinfo[-1] == '\n':
                sinfo = sinfo[:-1]
            sio.close()
        rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
        return rv

class logger:
    loggerName = ''
    fileName = ''
    fileMode = 'w'
    isStdout = True
    logLevel = logging.DEBUG
    stdoutLevel = logging.DEBUG
    _logger = logging.getLogger('logger')
    _isSetup = False
    @classmethod
    def _parse_level(cls, level):
        if isinstance(level, str):
            if level.isdigit():
                return int(level)
            elif hasattr(logging, level):
                return getattr(logging, level)
        return level

    @classmethod
    def setup(cls, **kwargs):
        if cls._isSetup:
            cls._logger.debug('I am already setup!')
            return cls._logger
        logging.setLoggerClass(WrappedLogger)
        cls.fileName = kwargs.get('filename', __name__ + '.log')
        cls.fileMode = kwargs.get('filemode', "w")
        cls.logLevel = cls._parse_level(kwargs.get('log_level', logging.DEBUG))
        cls.stdoutLevel =  cls._parse_level(kwargs.get('stdout_level', logging.DEBUG))
        cls.isStdout = kwargs.get('stdout', True)
        cls.loggerName = kwargs.get('logger', __name__)
        rootLogger = logging.getLogger()
        rootLogger.setLevel(cls._parse_level(kwargs.get('root_level', logging.DEBUG)))  # Set root logger level
        cls._logger = logging.getLogger(cls.loggerName)
        if not len(cls._logger.handlers):
            if cls.fileName:
                ffh = logging.Formatter(
                    '[%(levelname)s|%(asctime)s]]%(name)s: %(message)s',
                    '%d-%m-%Y %H:%M:%S'
                )
                #fh = logging.FileHandler(cls.fileName, mode=cls.fileMode)
                fh = RotatingFileHandler(
                    cls.fileName,
                    mode=cls.fileMode,
                    maxBytes=5000000
                )
                fh.setFormatter(ffh)
                fh.setLevel(cls.logLevel)
                cls._logger.addHandler(fh)

            if cls.isStdout:
                fsh = logging.Formatter(
                    '[%(levelname)s|%(asctime)s]]%(name)s: %(message)s',
                    '%d-%m-%Y %H:%M:%S'
                )
                sh = logging.StreamHandler(sys.stdout)
                sh.setFormatter(fsh)
                sh.setLevel(cls.stdoutLevel)
                cls._logger.addHandler(sh)
        #cls._logger.debug("LogLevel Debug")
        #cls._logger.info("LogLevel Info")
        #cls._logger.warning("LogLevel Warning")
        #cls._logger.error("LogLevel Error")
        #cls._logger.exception("LogLevel Exception")
        cls._isSetup = True
        return cls._logger

    @classmethod
    def get_logger(cls):
        return cls._logger

    @classmethod
    def with_logger(cls, func):
        @wraps(func)
        def decorator(*args, **kwargs):
            kwargs.update({'logger', cls.get_logger()})
            return func(*args, **kwargs)
        return decorator

    @classmethod
    def log_exceptions(cls, func):
        @wraps(func)
        def decorator(*args, **kwargs):
            result = None
            try:
                result = func(*args, **kwargs)
            except:
                err = "There was an exception in  "
                err += func.__name__
                cls._logger.exception(err)
                raise
            finally:
                return result
        return decorator

    @classmethod
    def debug(cls, *args, **kwargs):
        cls._logger.debug(*args, *kwargs)

    @classmethod
    def info(cls, *args, **kwargs):
        cls._logger.info(*args, *kwargs)

    @classmethod
    def warning(cls, *args, **kwargs):
        cls._logger.warning(*args, *kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        cls._logger.error(*args, *kwargs)

    @classmethod
    def exception(cls, *args, **kwargs):
        cls._logger.exception(*args, *kwargs)

    @classmethod
    def progressbar(cls, **kwargs):
        manager = enlighten.get_manager()
        return manager.counter(**kwargs)