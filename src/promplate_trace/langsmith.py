from datetime import datetime
from typing import Callable, Literal, Sequence

from langsmith import Client, RunTree
from promplate.llm.base import *
from promplate.prompt.chat import Message, assistant, ensure

from .env import env
from .utils import cache, wraps

RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


@cache
def get_client():
    return Client(env.langchain_endpoint, api_key=env.langchain_api_key)


def plant(
    name: str,
    run_type: RunType,
    inputs: dict,
    extra: dict | None = None,
    tags: Sequence[str] = (),
    error: str | None = None,
    outputs: dict | None = None,
    parent_run: RunTree | None = None,
):
    if parent_run:
        return parent_run.create_child(
            name,
            run_type,
            inputs=inputs,
            extra=extra,
            tags=tags,
            error=error,
            outputs=outputs,
        )
    return RunTree(
        name=name,
        run_type=run_type,
        inputs=inputs,
        extra=extra or {},
        tags=tags,
        error=error,
        outputs=outputs,
        project_name=env.langchain_project,
        client=get_client(),
    )


def plant_text_completions(
    function: Callable,
    text: str,
    config: dict,
    extra: dict | None = None,
    tags: Sequence[str] = (),
    error: str | None = None,
    outputs: dict | None = None,
    parent_run: RunTree | None = None,
):
    cls = function.__class__
    name = f"{cls.__module__}.{cls.__name__}"
    extra = extra or {} | {"invocation_params": config}
    return plant(name, "llm", {"prompt": text, **config}, extra, tags, error, outputs, parent_run)


def plant_chat_completions(
    function: Callable,
    messages: list[Message],
    config: dict,
    extra: dict | None = None,
    tags: Sequence[str] = (),
    error: str | None = None,
    outputs: dict | None = None,
    parent_run: RunTree | None = None,
):
    cls = function.__class__
    name = f"{cls.__module__}.{cls.__name__}"
    extra = extra or {} | {"invocation_params": config}
    return plant(
        name, "llm", {"messages": messages, **config}, extra, tags, error, outputs, parent_run
    )


def text_output(text=""):
    return {"choices": [{"text": text}]}


def chat_output(text=""):
    return {"choices": [{"message": assistant > text}]}


class patch:
    class text:
        @staticmethod
        def complete(f: Complete):
            @wraps(f)
            def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, outputs=text_output())
                run.post()
                out = f(text, **config)
                run.end(outputs=text_output(out))
                run.patch()
                return out

            return wrapper

        @staticmethod
        def acomplete(f: AsyncComplete):
            @wraps(f)
            async def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, outputs=text_output())
                run.post()
                out = await f(text, **config)
                run.end(outputs=text_output(out))
                run.patch()
                return out

            return wrapper

        @staticmethod
        def generate(f: Generate):
            @wraps(f)
            def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, outputs=text_output())
                run.post()
                out = ""
                for delta in f(text, **config):
                    if not out:
                        run.events = [{"name": "new_token", "time": datetime.utcnow()}]
                        run.post()
                    out += delta
                    yield delta
                run.end(outputs=text_output(out))
                run.patch()

            return wrapper

        @staticmethod
        def agenerate(f: AsyncGenerate):
            @wraps(f)
            async def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, outputs=text_output())
                run.post()
                out = ""
                async for delta in f(text, **config):
                    if not out:
                        run.events = [{"name": "new_token", "time": datetime.utcnow()}]
                        run.post()
                    out += delta
                    yield delta
                run.end(outputs=text_output(out))
                run.patch()

            return wrapper

    class chat:
        @staticmethod
        def complete(f: Complete):
            @wraps(f)
            def wrapper(messages: list[Message] | str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config)
                run.post()
                out = f(messages, **config)
                run.end(outputs=chat_output(out))
                run.patch()
                return out

            return wrapper

        @staticmethod
        def acomplete(f: AsyncComplete):
            @wraps(f)
            async def wrapper(messages: list[Message] | str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config)
                run.post()
                out = await f(messages, **config)
                run.end(outputs=chat_output(out))
                run.patch()
                return out

            return wrapper

        @staticmethod
        def generate(f: Generate):
            @wraps(f)
            def wrapper(messages: str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config)
                run.post()
                out = ""
                for delta in f(messages, **config):
                    if not out:
                        run.events = [{"name": "new_token", "time": datetime.utcnow()}]
                        run.post()
                    out += delta
                    yield delta
                run.end(outputs=chat_output(out))
                run.patch()

            return wrapper

        @staticmethod
        def agenerate(f: AsyncGenerate):
            @wraps(f)
            async def wrapper(messages: str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config)
                run.post()
                out = ""
                async for delta in f(messages, **config):
                    if not out:
                        run.events = [{"name": "new_token", "time": datetime.utcnow()}]
                        run.post()
                    out += delta
                    yield delta
                run.end(outputs=chat_output(out))
                run.patch()

            return wrapper
