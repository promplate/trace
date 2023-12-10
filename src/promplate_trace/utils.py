from datetime import datetime, timezone
from functools import wraps as _wraps
from importlib.metadata import version
from json import JSONEncoder, dumps, loads
from sys import version as python_version
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
