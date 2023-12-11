from datetime import datetime, timezone
from functools import wraps as _wraps
from importlib.metadata import version
from inspect import isfunction, ismethod
from json import JSONEncoder, dumps, loads
from sys import version as python_version
from typing import Callable, Hashable, ParamSpec, TypeVar, cast

from promplate import Context

P = ParamSpec("P")
T = TypeVar("T")


def wraps(target: Callable[P, T]) -> Callable[..., Callable[P, T]]:
    return _wraps(target)  # type: ignore


def cache(function: Callable[P, T]) -> Callable[P, T]:
    results: dict[Hashable, T] = {}

    @wraps(function)
    def wrapper(*args: Hashable):
        if args in results:
            return results[args]
        result = results[args] = function(*args)  # type: ignore
        return result

    return wrapper


def only_once(decorator: Callable[P, T]) -> Callable[P, T]:
    @wraps(decorator)
    def wrapper(function):
        decorators = getattr(function, "__decorators__", [])
        if decorator not in decorators:
            function = decorator(function)  # type: ignore
            function.__decorators__ = decorators + [decorator]

        return function

    return cast(T, wrapper)  # type: ignore


def diff_context(context_in: Context, context_out: Context) -> Context:
    return {k: v for k, v in context_out.items() if k not in context_in or context_in[k] != v}


def get_versions(*packages: str):
    return {package: version(package) for package in packages} | {"python": python_version}


def utcnow():
    return datetime.now(timezone.utc)


class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        if hasattr(o, "model_dump_json") and callable(o.model_dump_json):
            return o.model_dump_json()
        if hasattr(o, "json") and callable(o.json):
            return o.json()
        try:
            return super().default(o)
        except TypeError:
            return repr(o)


def ensure_serializable(context: Context):
    return loads(dumps(context, ensure_ascii=False, cls=CustomJSONEncoder))


def name(function: Callable):
    if isfunction(function):
        return f"{function.__module__}.{function.__name__}"
    cls = (function.__self__ if ismethod(function) else function).__class__
    return f"{cls.__module__}.{cls.__name__}"
