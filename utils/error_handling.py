import functools
import logging

import pandas as pd


def get_primitive_args(args):
    res = []
    for arg in args:
        if isinstance(arg, (int, float, str, bool)):
            res.append(arg)
        elif isinstance(arg, pd.Series):
            res.append(arg.values)
        elif arg is None:
            res.append("None")
        else:
            res.append("non-primitive arg")

    return res


def log_func_info(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(
            f"Func: {func.__name__} | Args: {get_primitive_args(args)} | Kwargs: {get_primitive_args(kwargs)}"
        )
        return func(*args, **kwargs)

    return wrapper


def trycatch_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(
                f"Error in {func.__name__} : Args: {get_primitive_args(args)} | Kwargs: {get_primitive_args(kwargs)} | Error: {e}"
            )
            return None

    return wrapper
