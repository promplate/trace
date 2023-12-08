from datetime import datetime
from typing import AsyncIterable, Callable, Iterable, Literal, Sequence, cast

from langsmith import Client, RunTree
from promplate.chain.node import AbstractChain, Chain, ChainContext, JumpTo, Node
from promplate.llm.base import AsyncComplete, AsyncGenerate, Complete, Generate
from promplate.prompt.chat import Message, assistant, ensure
from promplate.prompt.template import Context

from .env import env
from .utils import cache, diff_context, wraps

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
            tags=list(tags),
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
    return plant(name, "llm", {"messages": messages, **config}, extra, tags, error, outputs, parent_run)


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
                run = plant_text_completions(f, text, config, outputs=text_output(), parent_run=config.pop("__parent__", None))
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
                run = plant_text_completions(f, text, config, outputs=text_output(), parent_run=config.pop("__parent__", None))
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
                run = plant_text_completions(f, text, config, outputs=text_output(), parent_run=config.pop("__parent__", None))
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
                run = plant_text_completions(f, text, config, outputs=text_output(), parent_run=config.pop("__parent__", None))
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
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop("__parent__", None))
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
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop("__parent__", None))
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
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop("__parent__", None))
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
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop("__parent__", None))
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

    @staticmethod
    def chain(ChainClass: type[Chain]):
        class TraceableNode(ChainClass):
            def on_chain_start(self, context: Context | None = None, **config):
                context_in = {} if context is None else {k: v for k, v in context.items() if k != "__parent__"}
                run = plant(str(self), "chain", context_in, parent_run=config.get("__parent__"))
                context_out = ChainContext.ensure(context)
                context_out["__parent__"] = config["__parent__"] = run
                run.post()
                return run, context_in, context_out, config

            def on_chain_end(self, run: RunTree, config, context_in, context_out):
                run.end(outputs=diff_context(context_in, context_out))
                run.patch()
                config["__parent__"] = run.parent_run
                return config

            def invoke(self, context=None, /, complete=None, **config) -> ChainContext:
                run, context_in, context, config = self.on_chain_start(context, **config)

                try:
                    self._invoke(ChainContext(context, self.context), complete, **config)
                    self.on_chain_end(run, config, context_in, context)
                except JumpTo as jump:
                    config = self.on_chain_end(run, config, context_in, context)
                    if jump.target is None or jump.target is self:
                        jump.chain.invoke(context, complete, **config)
                    else:
                        raise jump from None

                return context

            async def ainvoke(self, context=None, /, complete=None, **config) -> ChainContext:
                run, context_in, context, config = self.on_chain_start(context, **config)

                try:
                    await self._ainvoke(ChainContext(context, self.context), complete, **config)
                    self.on_chain_end(run, config, context_in, context)
                except JumpTo as jump:
                    config = self.on_chain_end(run, config, context_in, context)
                    if jump.target is None or jump.target is self:
                        await jump.chain.ainvoke(context, complete, **config)
                    else:
                        raise jump from None

                return context

            def stream(self, context=None, /, generate=None, **config) -> Iterable[ChainContext]:
                run, context_in, context, config = self.on_chain_start(context, **config)

                try:
                    for _ in self._stream(ChainContext(context, self.context), generate, **config):
                        yield context
                    self.on_chain_end(run, config, context_in, context)
                except JumpTo as jump:
                    config = self.on_chain_end(run, config, context_in, context)
                    if jump.target is None or jump.target is self:
                        yield from jump.chain.stream(context, generate, **config)
                    else:
                        raise jump from None

            async def astream(self, context=None, /, generate=None, **config) -> AsyncIterable[ChainContext]:
                run, context_in, context, config = self.on_chain_start(context, **config)

                try:
                    async for _ in self._astream(ChainContext(context, self.context), generate, **config):
                        yield context
                    self.on_chain_end(run, config, context_in, context)
                except JumpTo as jump:
                    config = self.on_chain_end(run, config, context_in, context)
                    if jump.target is None or jump.target is self:
                        async for i in jump.chain.astream(context, generate, **config):
                            yield i
                    else:
                        raise jump from None

        return TraceableNode

    @staticmethod
    def node(NodeClass: type[Node]):
        class TraceableChain(cast(type[Node], patch.chain(NodeClass))):  # type: ignore
            Chain = patch.chain(Chain)

            def next(self, chain: AbstractChain):
                if isinstance(chain, Chain):
                    return self.Chain(self, *chain)
                else:
                    return self.Chain(self, chain)

            def render(self, context: Context | None = None):
                context = ChainContext(context, self.context)
                parent_run = context.pop("__parent__", None)
                self._apply_pre_processes(context)
                run = plant(
                    "render",
                    "prompt",
                    {
                        "template": self.template.text,
                        "context": {} if context is None else {**context},
                    },
                    parent_run=parent_run,
                )
                prompt = self.template.render(context)
                run.end(outputs={"output": prompt})
                run.post()
                return prompt

            async def arender(self, context: Context | None = None):
                context = ChainContext(context, self.context)
                parent_run = context.pop("__parent__", None)
                await self._apply_async_pre_processes(context)
                run = plant(
                    "arender",
                    "prompt",
                    {
                        "template": self.template.text,
                        "context": {} if context is None else {**context},
                    },
                    parent_run=parent_run,
                )
                prompt = await self.template.arender(context)
                run.end(outputs={"output": prompt})
                run.post()
                return prompt

        return TraceableChain
