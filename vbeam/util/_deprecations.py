import functools
import warnings


def _get_function_name(f: callable):
    return getattr(f, "__qualname__", getattr(f, "__name__"))


def deprecated(version: str, reason):
    """Warn when calling f because it is deprecated.

    >>> import warnings
    >>> @deprecated("1.0.6", "Use new_function() instead.")
    ... def f(a):
    ...   return a + 1
    >>> with warnings.catch_warnings(record=True) as w:
    ...   f(23)
    ...   print(w[0].message)
    24
    f is deprecated since version 1.0.6. Use new_function() instead.
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{f.__name__} is deprecated since version {version}. {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return f(*args, **kwargs)

        return wrapper

    return decorator


def renamed_kwargs(version: str, **renamed_kwargs: str):
    """Warn when passing an argument by name when that argument has been renamed in a
    recent version and the old name is deprecated.

    >>> import warnings
    >>> @renamed_kwargs("1.0.5", b="renamed_arg")
    ... def f(a, renamed_arg):
    ...   return a + renamed_arg
    >>> with warnings.catch_warnings(record=True) as w:
    ...   f(a=1, b=2)
    ...   print(w[0].message)
    3
    Deprecation warning: argument 'b' of f was renamed to 'renamed_arg' in version 1.0.5.

    Using the new name or given positional arguments doesn't print the warning.
    >>> f(a=1, renamed_arg=2)
    3
    >>> f(1, 2)
    3
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            new_kwargs = {}
            for k, v in kwargs.items():
                if k in renamed_kwargs:
                    warnings.warn(
                        f"Deprecation warning: argument '{k}' of "
                        f"{_get_function_name(f)} was renamed to '{renamed_kwargs[k]}' "
                        f"in version {version}.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    k = renamed_kwargs[k]
                new_kwargs[k] = v
            return f(*args, **new_kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    import doctest

    doctest.testmod()
