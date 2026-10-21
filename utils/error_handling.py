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


def pool_worker(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from utils.preprocessing import drain_unmapped_counts

        report = args[0].report
        previous_drops = report.drops.copy()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in pool worker {func.__name__}: {e}")
            report.record_drop("worker failed")
            result = None
        if not result and report.drops == previous_drops:
            report.record_drop("no usable exam")
        return result, drain_unmapped_counts(), report.drops - previous_drops

    return wrapper
