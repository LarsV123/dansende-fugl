from functools import wraps, partial
import cProfile
import pstats
from time import perf_counter
from tabulate import tabulate
import inspect


def time_this(f):
    """
    Simple decorator for timing a function.
    Prints total runtime after the function is done.
    """

    @wraps(f)
    def decorator(*args, **kwargs):
        start = perf_counter()
        result = f(*args, **kwargs)
        end = perf_counter()
        elapsed = round(end - start, 3)

        print()
        print(f"Run time for {f.__name__}: {elapsed} seconds")
        print()
        return result

    return decorator


def profile_this(f=None, *, verbose=True, output_path="results.prof"):
    """
    Simple decorator for profiling a function.

    Generates a logfile at the specified relative path, with detailed
    statistics which you can run with Snakeviz.

    Pass verbose=False to suppress detailed print statements.
    """
    if f is None:
        return partial(profile_this, verbose=verbose, output_path=output_path)

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        with cProfile.Profile() as pr:
            result = f(*args, **kwargs)
        end = perf_counter()
        elapsed = round(end - start, 3)

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.dump_stats(filename=output_path)

        print()
        print("*** Profiler information *** ")
        if verbose:
            stats.print_stats(10)

            print(f"Profiler details ({profile_this.__name__})")
            header = ["Key", "Value"]
            table = [
                ("Function name", f.__name__),
                ("Verbose mode", verbose),
                ("Output path", output_path),
                ("Run time (seconds)", elapsed),
            ]
            print(tabulate(table, headers=header, tablefmt="psql"))
            print()

            print(f"Arguments passed to {f.__name__}")
            func_args = inspect.signature(f).bind(*args, **kwargs).arguments
            f_table = [(k, v) for k, v in func_args.items()]
            print(tabulate(f_table, headers=header, tablefmt="psql"))
            print()

        print(f"Run time for {f.__name__}: {elapsed} seconds")
        print(f"Inspect results with the command: snakeviz {output_path}")
        return result

    return wrapper
