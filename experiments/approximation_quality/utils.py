from functools import wraps


def unpack(func):
    @wraps(func)
    def my_func(args):
        return func(*args)

    return my_func
