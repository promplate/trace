from functools import wraps as _wraps
from typing import Callable, ParamSpec, TypeVar, cast

P = ParamSpec("P")
T = TypeVar("T")


def wraps(target: Callable[P, T]) -> Callable[..., Callable[P, T]]:
    return _wraps(target)


def cache(function: Callable[[], T]):
    result = None

    @wraps(function)
    def wrapper():
        nonlocal result
        if result is None:
            result = function()
        return result

    return cast(Callable[[], T], wrapper)
