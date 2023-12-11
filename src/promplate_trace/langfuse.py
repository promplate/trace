from typing import AsyncIterable, Callable, Iterable, cast

from langfuse import Langfuse
from langfuse.client import StatefulClient, StatefulSpanClient, StatefulTraceClient, StateType
from langfuse.model import CreateEvent, CreateGeneration, CreateSpan, CreateTrace, UpdateGeneration, UpdateSpan
from promplate.chain.node import AbstractChain, Chain, ChainContext, JumpTo, Node
from promplate.llm.base import AsyncComplete, AsyncGenerate, Complete, Generate
from promplate.prompt.chat import Message, assistant, ensure
from promplate.prompt.template import Context

from .env import env
from .utils import cache, diff_context, ensure_serializable, get_versions, only_once, utcnow, wraps


@cache
def get_client():
    return Langfuse(env.langfuse_public_key, env.langfuse_secret_key, env.langfuse_host)


@cache
def get_version_info():
    return " ".join(f"{k}/{v}" for k, v in get_versions("promplate", "promplate-trace").items())


def ensure_parent_run(parent: StatefulClient | str | None):
    if isinstance(parent, str):
        return StatefulTraceClient(get_client().client, parent, StateType.TRACE, parent, get_client().task_manager)

    metadata = get_versions("promplate", "promplate-trace", "langfuse")
    parent = parent or get_client().trace(CreateTrace(metadata=metadata))
    assert parent is not None
    return parent


def plant_text_completions(function: Callable, text: str, config: dict, parent_run: StatefulClient | None = None):
    cls = function.__class__
    name = f"{cls.__module__}.{cls.__name__}"
    parent = ensure_parent_run(parent_run)
    run = parent.generation(
        CreateGeneration(
            name=name,
            prompt=text,
            model=config.get("model", None),
            startTime=utcnow(),
            modelParameters={k: v for k, v in config.items() if k != "model"},
        )
    )
    assert run is not None
    return run


def plant_chat_completions(function: Callable, messages: list[Message], config: dict, parent_run: StatefulClient | None = None):
    cls = function.__class__
    name = f"{cls.__module__}.{cls.__name__}"
    parent = ensure_parent_run(parent_run)
    run = parent.generation(
        CreateGeneration(
            name=name,
            prompt=messages,
            model=config.get("model", None),
            startTime=utcnow(),
            modelParameters={k: v for k, v in config.items() if k != "model"},
        )
    )
    assert run is not None
    return run


class patch:
    class text:
        @staticmethod
        @only_once
        def complete(f: Complete):
            @wraps(f)
            def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, parent_run=config.pop("__lf_parent__", None))
                out = f(text, **config)
                run.end(UpdateGeneration(generationId=run.id, completion=out))
                return out

            return wrapper

        @staticmethod
        @only_once
        def acomplete(f: AsyncComplete):
            @wraps(f)
            async def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, parent_run=config.pop("__lf_parent__", None))
                out = await f(text, **config)
                run.end(UpdateGeneration(generationId=run.id, completion=out))
                return out

            return wrapper

        @staticmethod
        @only_once
        def generate(f: Generate):
            @wraps(f)
            def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, parent_run=config.pop("__lf_parent__", None))
                out = ""
                for delta in f(text, **config):
                    if not out:
                        run.update(UpdateGeneration(generationId=run.id, completionStartTime=utcnow()))
                    out += delta
                    yield delta
                run.end(UpdateGeneration(generationId=run.id, completion=out, endTime=utcnow()))

            return wrapper

        @staticmethod
        @only_once
        def agenerate(f: AsyncGenerate):
            @wraps(f)
            async def wrapper(text: str, /, **config):
                run = plant_text_completions(f, text, config, parent_run=config.pop("__lf_parent__", None))
                out = ""
                async for delta in f(text, **config):
                    if not out:
                        run.update(UpdateGeneration(generationId=run.id, completionStartTime=utcnow()))
                    out += delta
                    yield delta
                run.end(UpdateGeneration(generationId=run.id, completion=out, endTime=utcnow()))

            return wrapper

    class chat:
        @staticmethod
        @only_once
        def complete(f: Complete):
            @wraps(f)
            def wrapper(messages: list[Message] | str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop("__lf_parent__", None))
                out = f(messages, **config)
                run.end(UpdateGeneration(generationId=run.id, completion={"choices": [{"message": assistant > out}]}))
                return out

            return wrapper

        @staticmethod
        @only_once
        def acomplete(f: AsyncComplete):
            @wraps(f)
            async def wrapper(messages: list[Message] | str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop("__lf_parent__", None))
                out = await f(messages, **config)
                run.end(UpdateGeneration(generationId=run.id, completion={"choices": [{"message": assistant > out}]}))
                return out

            return wrapper

        @staticmethod
        @only_once
        def generate(f: Generate):
            @wraps(f)
            def wrapper(messages: str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop("__lf_parent__", None))
                out = ""
                for delta in f(messages, **config):
                    if not out:
                        run.update(UpdateGeneration(generationId=run.id, completionStartTime=utcnow()))
                    out += delta
                    yield delta
                run.end(UpdateGeneration(generationId=run.id, completion=out, endTime=utcnow()))

            return wrapper

        @staticmethod
        @only_once
        def agenerate(f: AsyncGenerate):
            @wraps(f)
            async def wrapper(messages: str, /, **config):
                run = plant_chat_completions(f, ensure(messages), config, parent_run=config.pop("__lf_parent__", None))
                out = ""
                async for delta in f(messages, **config):
                    if not out:
                        run.update(UpdateGeneration(generationId=run.id, completionStartTime=utcnow()))
                    out += delta
                    yield delta
                run.end(UpdateGeneration(generationId=run.id, completion=out, endTime=utcnow()))

            return wrapper

    @staticmethod
    @only_once
    def chain(ChainClass: type[Chain]):
        class TraceableChain(ChainClass):
            def on_chain_start(self, context: Context | None = None, **config):
                context_in = {} if context is None else {k: v for k, v in context.items() if not k.endswith("parent__")}
                run = ensure_parent_run(config.get("__lf_parent__")).span(
                    CreateSpan(
                        name=str(self),
                        input=ensure_serializable(context_in),
                        startTime=utcnow(),
                    )
                )
                assert run is not None
                context_out = ChainContext.ensure(context)
                context_out["__lf_parent__"] = config["__lf_parent__"] = run
                return run, context_in, context_out, config

            def on_chain_end(self, run: StatefulSpanClient, config, context_in, context_out):
                run.end(
                    UpdateSpan(
                        spanId=run.id,
                        output=ensure_serializable(diff_context(context_in, context_out)),
                        endTime=utcnow(),
                    )
                )
                return config

            def invoke(self, context=None, /, complete=None, **config) -> ChainContext:
                parent_run = config.get("__lf_parent__")
                run, context_in, context, config = self.on_chain_start(context, **config)

                try:
                    self._invoke(ChainContext(context, self.context), complete, **config)
                    self.on_chain_end(run, config, context_in, context)
                except JumpTo as jump:
                    config = self.on_chain_end(run, config, context_in, context)
                    config["__lf_parent__"] = parent_run
                    if jump.target is None or jump.target is self:
                        jump.chain.invoke(context, complete, **config)
                    else:
                        raise jump from None

                return context

            async def ainvoke(self, context=None, /, complete=None, **config) -> ChainContext:
                parent_run = config.get("__lf_parent__")
                run, context_in, context, config = self.on_chain_start(context, **config)

                try:
                    await self._ainvoke(ChainContext(context, self.context), complete, **config)
                    self.on_chain_end(run, config, context_in, context)
                except JumpTo as jump:
                    config = self.on_chain_end(run, config, context_in, context)
                    config["__lf_parent__"] = parent_run
                    if jump.target is None or jump.target is self:
                        await jump.chain.ainvoke(context, complete, **config)
                    else:
                        raise jump from None

                return context

            def stream(self, context=None, /, generate=None, **config) -> Iterable[ChainContext]:
                parent_run = config.get("__lf_parent__")
                run, context_in, context, config = self.on_chain_start(context, **config)

                try:
                    for _ in self._stream(ChainContext(context, self.context), generate, **config):
                        yield context
                    self.on_chain_end(run, config, context_in, context)
                except JumpTo as jump:
                    config = self.on_chain_end(run, config, context_in, context)
                    config["__lf_parent__"] = parent_run
                    if jump.target is None or jump.target is self:
                        yield from jump.chain.stream(context, generate, **config)
                    else:
                        raise jump from None

            async def astream(self, context=None, /, generate=None, **config) -> AsyncIterable[ChainContext]:
                parent_run = config.get("__lf_parent__")
                run, context_in, context, config = self.on_chain_start(context, **config)

                try:
                    async for _ in self._astream(ChainContext(context, self.context), generate, **config):
                        yield context
                    self.on_chain_end(run, config, context_in, context)
                except JumpTo as jump:
                    config = self.on_chain_end(run, config, context_in, context)
                    config["__lf_parent__"] = parent_run
                    if jump.target is None or jump.target is self:
                        async for i in jump.chain.astream(context, generate, **config):
                            yield i
                    else:
                        raise jump from None

            def next(self, chain: AbstractChain):
                if isinstance(chain, Node):
                    return patch.chain(Chain)(*self, chain)
                elif isinstance(chain, Chain):
                    return patch.chain(Chain)(*self, *chain)
                else:
                    raise NotImplementedError

        return TraceableChain

    @staticmethod
    @only_once
    def node(NodeClass: type[Node]):
        class TraceableNode(cast(type[Node], patch.chain(NodeClass))):  # type: ignore
            def next(self, chain: AbstractChain):
                if isinstance(chain, Chain):
                    return patch.chain(Chain)(self, *chain)
                else:
                    return patch.chain(Chain)(self, chain)

            def render(self, context: Context | None = None):
                context = ChainContext(context, self.context)
                parent_run = context.pop("__lf_parent__", None)
                self._apply_pre_processes(context)
                prompt = self.template.render(context)
                ensure_parent_run(parent_run).event(
                    CreateEvent(
                        name="render",
                        input={"template": self.template.text, "context": {} if context is None else ensure_serializable({**context})},
                        output=prompt,
                        startTime=utcnow(),
                    )
                )
                return prompt

            async def arender(self, context: Context | None = None):
                context = ChainContext(context, self.context)
                parent_run = context.pop("__lf_parent__", None)
                await self._apply_async_pre_processes(context)
                prompt = await self.template.arender(context)
                ensure_parent_run(parent_run).event(
                    CreateEvent(
                        name="render",
                        input={"template": self.template.text, "context": {} if context is None else ensure_serializable({**context})},
                        output=prompt,
                        startTime=utcnow(),
                    )
                )
                return prompt

        return TraceableNode
