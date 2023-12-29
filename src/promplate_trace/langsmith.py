from typing import Callable, Literal, Sequence, TypeVar, cast

from langsmith import Client, RunTree
from promplate.chain.node import BaseCallback, Chain, Interruptable, Node
from promplate.llm.base import AsyncComplete, AsyncGenerate, Complete, Generate
from promplate.prompt import Context
from promplate.prompt.chat import Message, assistant, ensure
from promplate.prompt.template import Context

from .env import env
from .utils import cache, clean, diff_context, get_versions, name, only_once, utcnow, wraps

RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]

LS_PARENT = "__ls_parent__"


def find_run(*contexts: Context) -> RunTree | None:
    for context in contexts:
        if run := context.get(LS_PARENT):
            return run


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
            tags=list(tags),
            error=error,
            outputs=outputs,
        )
    return RunTree(
        name=name,
        run_type=run_type,
        inputs=inputs,
        extra=(extra or {}) | {"runtime": get_versions("promplate", "promplate-trace", "langsmith")},
        tags=tags,
        error=error,
        outputs=outputs,
        project_name=env.langchain_project,
        client=get_client(),
    )


def plant_text_completions(
    function: Callable,
    text: str,
    config: Context,
    extra: dict | None = None,
    tags: Sequence[str] = (),
    error: str | None = None,
    outputs: dict | None = None,
    parent_run: RunTree | None = None,
):
    config = clean(config)
    extra = (extra or {}) | {"invocation_params": config}
    return plant(name(function), "llm", {"prompt": text, **config}, extra, tags, error, outputs, parent_run)


def plant_chat_completions(
    function: Callable,
    messages: list[Message],
    config: Context,
    extra: dict | None = None,
    tags: Sequence[str] = (),
    error: str | None = None,
    outputs: dict | None = None,
    parent_run: RunTree | None = None,
):
    config = clean(config)
    extra = (extra or {}) | {"invocation_params": config}
    return plant(name(function), "llm", {"messages": messages, **config}, extra, tags, error, outputs, parent_run)


def text_output(text=""):
    return {"choices": [{"text": text}]}


def chat_output(text=""):
    return {"choices": [{"message": assistant > text}]}


class TraceCallback(BaseCallback):
    def on_enter(self, node, context: Context | None, config: Context):
        context_in = self.context_in = {} if context is None else clean(context)

        parent_run = find_run(config, context_in)

        run = self.run = plant(str(node), "chain", context_in, parent_run=parent_run)
        run.post()

        if context is None:
            context = {}

        context[LS_PARENT] = config[LS_PARENT] = run

        return context, config

    def on_leave(self, _, context: Context, config: Context):  # type: ignore
        context_out = clean(context)

        self.run.end(outputs=diff_context(self.context_in, context_out))
        self.run.patch()

        context[LS_PARENT] = config[LS_PARENT] = self.run.parent_run

        return context, config


T = TypeVar("T", bound=Interruptable)


class patch:
    class text:
        @staticmethod
        @only_once
        def complete(f: Complete):
            @wraps(f)
            def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, outputs=text_output(), parent_run=config.pop(LS_PARENT, None))
                run.post()
                out = f(text, **config)
                run.end(outputs=text_output(out))
                run.patch()
                return out

            return wrapper

        @staticmethod
        @only_once
        def acomplete(f: AsyncComplete):
            @wraps(f)
            async def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, outputs=text_output(), parent_run=config.pop(LS_PARENT, None))
                run.post()
                out = await f(text, **config)
                run.end(outputs=text_output(out))
                run.patch()
                return out

            return wrapper

        @staticmethod
        @only_once
        def generate(f: Generate):
            @wraps(f)
            def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, outputs=text_output(), parent_run=config.pop(LS_PARENT, None))
                run.post()
                out = ""
                for delta in f(text, **config):
                    if not out:
                        run.events = [{"name": "new_token", "time": utcnow()}]
                        run.post()
                    out += delta
                    yield delta
                run.end(outputs=text_output(out))
                run.patch()

            return wrapper

        @staticmethod
        @only_once
        def agenerate(f: AsyncGenerate):
            @wraps(f)
            async def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, outputs=text_output(), parent_run=config.pop(LS_PARENT, None))
                run.post()
                out = ""
                async for delta in f(text, **config):
                    if not out:
                        run.events = [{"name": "new_token", "time": utcnow()}]
                        run.post()
                    out += delta
                    yield delta
                run.end(outputs=text_output(out))
                run.patch()

            return wrapper

    class chat:
        @staticmethod
        @only_once
        def complete(f: Complete):
            @wraps(f)
            def wrapper(messages: list[Message] | str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop(LS_PARENT, None))
                run.post()
                out = f(messages, **config)
                run.end(outputs=chat_output(out))
                run.patch()
                return out

            return wrapper

        @staticmethod
        @only_once
        def acomplete(f: AsyncComplete):
            @wraps(f)
            async def wrapper(messages: list[Message] | str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop(LS_PARENT, None))
                run.post()
                out = await f(messages, **config)
                run.end(outputs=chat_output(out))
                run.patch()
                return out

            return wrapper

        @staticmethod
        @only_once
        def generate(f: Generate):
            @wraps(f)
            def wrapper(messages: str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop(LS_PARENT, None))
                run.post()
                out = ""
                for delta in f(messages, **config):
                    if not out:
                        run.events = [{"name": "new_token", "time": utcnow()}]
                        run.post()
                    out += delta
                    yield delta
                run.end(outputs=chat_output(out))
                run.patch()

            return wrapper

        @staticmethod
        @only_once
        def agenerate(f: AsyncGenerate):
            @wraps(f)
            async def wrapper(messages: str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop(LS_PARENT, None))
                run.post()
                out = ""
                async for delta in f(messages, **config):
                    if not out:
                        run.events = [{"name": "new_token", "time": utcnow()}]
                        run.post()
                    out += delta
                    yield delta
                run.end(outputs=chat_output(out))
                run.patch()

            return wrapper

    @staticmethod
    @only_once
    def chain(ChainClass: type[T]):
        class TraceableChain(cast(type[Chain], ChainClass)):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.callbacks.append(TraceCallback)

        return cast(type[T], TraceableChain)

    @staticmethod
    @only_once
    def node(NodeClass: type[Node]):
        class TraceableNode(patch.chain(NodeClass)):
            def _get_chain_type(self):  # type: ignore
                return patch.chain(super()._get_chain_type())

            def render(self, context: Context | None = None, callbacks=None):
                parent_run = None if context is None else find_run(context)

                prompt = super().render(context, callbacks)

                run = plant(
                    "render",
                    "prompt",
                    {
                        "template": self.template.text,
                        "context": {} if context is None else clean(context),
                    },
                    parent_run=parent_run,
                )
                run.end(outputs={"output": prompt})
                run.post()

                return prompt

            async def arender(self, context: Context | None = None, callbacks=None):
                parent_run = None if context is None else find_run(context)

                prompt = await super().arender(context, callbacks)

                run = plant(
                    "arender",
                    "prompt",
                    {
                        "template": self.template.text,
                        "context": {} if context is None else clean(context),
                    },
                    parent_run=parent_run,
                )
                run.end(outputs={"output": prompt})
                run.post()

                return prompt

        return TraceableNode
