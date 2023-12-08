from functools import wraps as _wraps
from typing import Callable, ParamSpec, TypeVar

from promplate import Context

P = ParamSpec("P")
T = TypeVar("T")


def wraps(target: Callable[P, T]) -> Callable[..., Callable[P, T]]:
    return _wraps(target)  # type: ignore


def cache(function: Callable[[], T]) -> Callable[[], T]:
    result = None

    @wraps(function)
    def wrapper():
        nonlocal result
        if result is None:
            result = function()
        return result

    return wrapper


def diff_context(context_in: Context, context_out: Context) -> Context:
    return {k: v for k, v in context_out.items() if k not in context_in or context_in[k] != v}
