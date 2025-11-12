from functools import wraps
from time import perf_counter
from time import strftime
from time import gmtime


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = perf_counter()
        result = func(*args, **kargs)
        elapsed_time = perf_counter() - start
        end = strftime("%H:%M:%S", gmtime(elapsed_time))
        print("[{}] elapsed time: {}".format(func.__name__, end))
        return result

    return wrapper
