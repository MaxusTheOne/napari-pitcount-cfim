import logging
import sys
import threading
from qtpy.QtCore import qInstallMessageHandler, QtMsgType

def setup_python_logging():
    # Root logger to stdout (and optionally to file)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s"
    )
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    # Optionally file handler
    # fh = logging.FileHandler("app_debug.log")
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(fmt)
    # root.addHandler(fh)

def qt_message_logger(msg_type: QtMsgType, context, message: str):
    # Map Qt message types to logging levels
    level = {
        QtMsgType.QtDebugMsg:     logging.DEBUG,
        QtMsgType.QtInfoMsg:      logging.INFO,
        QtMsgType.QtWarningMsg:   logging.WARNING,
        QtMsgType.QtCriticalMsg:  logging.ERROR,
        QtMsgType.QtFatalMsg:     logging.CRITICAL,
    }.get(msg_type, logging.INFO)
    logging.log(level, f"Qt: {message} ({context.file}:{context.line})")

def setup_thread_exception_hook():
    # Python 3.8+ will call this for uncaught exceptions in threads
    def handle_thread_exc(args: threading.ExceptHookArgs):
        logging.error(
            f"Uncaught exception in thread {args.thread.name!r}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )
    threading.excepthook = handle_thread_exc


