from inspect import isasyncgen, isawaitable, isgenerator
from typing import Callable, TypeVar, cast

from langfuse import Langfuse
from langfuse.client import StatefulClient, StatefulGenerationClient, StatefulSpanClient, StatefulTraceClient, StateType
from promplate.chain.node import BaseCallback, Chain, Interruptable, Node
from promplate.llm.base import LLM, AsyncComplete, AsyncGenerate, Complete, Generate
from promplate.prompt.chat import Message, ensure
from promplate.prompt.template import Context

from .env import env
from .utils import as_is_decorator, cache, clean, diff_context, ensure_serializable, get_versions, name, only_once, split_model_parameters, utcnow, wraps

MaybeRun = StatefulClient | str | None

LF_PARENT = "__lf_parent__"


def find_runs(*contexts: Context) -> list[MaybeRun]:
    for context in contexts:
        if run_stack := context.get(LF_PARENT):
            return run_stack

    return [None]


@cache
def get_client():
    return Langfuse(env.langfuse_public_key, env.langfuse_secret_key, env.langfuse_host)


def ensure_parent_run(parent: MaybeRun):
    if isinstance(parent, str):
        return StatefulTraceClient(get_client().client, parent, StateType.TRACE, parent, get_client().task_manager)

    metadata = get_versions("promplate", "promplate-trace", "langfuse")
    parent = parent or get_client().trace(metadata=metadata)
    assert parent is not None
    return parent


def plant_text_completions(function: Callable, text: str, config: Context, parent_run: MaybeRun = None):
    parent = ensure_parent_run(parent_run)
    config, extras = split_model_parameters(config)
    run = parent.generation(
        name=name(function),
        input=text,
        model=str(config.pop("model", None)),
        start_time=utcnow(),
        model_parameters=config,
        metadata=extras,
    )
    assert run is not None
    return run


def plant_chat_completions(function: Callable, messages: str | list[Message], config: Context, parent_run: MaybeRun = None):
    parent = ensure_parent_run(parent_run)
    config, extras = split_model_parameters(config)
    run = parent.generation(
        name=name(function),
        input=ensure(messages),
        model=str(config.pop("model", None)),
        start_time=utcnow(),
        model_parameters=config,
        metadata=extras,
    )
    assert run is not None
    return run


class TraceCallback(BaseCallback):
    def on_enter(self, node, context: Context | None, config: Context):
        context_in = self.context_in = {} if context is None else clean(context)

        runs = self.runs = find_runs(config, context_in)
        parent_run = ensure_parent_run(runs[-1])

        run = parent_run.span(name=str(node), input=ensure_serializable(context_in), start_time=utcnow())

        if context is None:
            context = {}

        context[LF_PARENT] = config[LF_PARENT] = [*runs, run]

        return context, config

    def on_leave(self, _, context: Context, config: Context):  # type: ignore
        context_out = clean(context)

        runs = find_runs(config, context)
        run = cast(StatefulSpanClient, runs[-1])

        run.end(output=ensure_serializable(diff_context(self.context_in, context_out)), end_time=utcnow())

        context[LF_PARENT] = config[LF_PARENT] = runs[:-1]

        return context, config


T = TypeVar("T", bound=Interruptable)


class patch:
    @staticmethod
    def _make_complete_wrapper(plant: Callable[..., StatefulGenerationClient], original):
        @wraps(original)
        def wrapper(prompt, /, **config):
            run = plant(original, prompt, config, parent_run=config.pop(LF_PARENT, [None])[-1])
            out = original(prompt, **config)

            if isawaitable(out):

                async def _():
                    result = await out
                    run.end(output=result)
                    return result

                return _()

            run.end(output=out)
            return out

        return wrapper

    @staticmethod
    def _make_generate_wrapper(plant: Callable[..., StatefulGenerationClient], original):
        @wraps(original)
        def wrapper(prompt, /, **config):
            run = plant(original, prompt, config, parent_run=config.pop(LF_PARENT, [None])[-1])
            gen = original(prompt, **config)

            if isasyncgen(gen):

                async def _():
                    out = ""
                    async for delta in gen:
                        if not out:
                            run.update(completion_start_time=utcnow())
                        out += delta
                        yield delta
                    run.end(output=out, end_time=utcnow())

                return _()

            assert isgenerator(gen)

            def _():
                out = ""
                for delta in gen:
                    if not out:
                        run.update(completion_start_time=utcnow())
                    out += delta
                    yield delta
                run.end(output=out, end_time=utcnow())

            return _()

        return wrapper

    @staticmethod
    def _make_dispatch_wrapper(plant: Callable[..., StatefulGenerationClient], original):
        @wraps(original)
        def wrapper(prompt, /, **config):
            run = plant(original, prompt, config, config.pop(LF_PARENT, [None])[-1])
            res = original(prompt, **config)

            if isasyncgen(res):

                async def _():
                    out = ""
                    async for delta in res:
                        if not out:
                            run.update(completion_start_time=utcnow())
                        out += delta
                        yield delta
                    run.end(output=out, end_time=utcnow())

                return _()

            if isawaitable(res):

                async def _():
                    result = await res
                    run.end(output=result)
                    return result

                return _()

            if isinstance(res, str):
                run.end(output=res, end_time=utcnow())
                return res

            assert isgenerator(res)

            def _():
                out = ""
                for delta in res:
                    if not out:
                        run.update(completion_start_time=utcnow())
                    out += delta
                    yield delta
                run.end(output=out, end_time=utcnow())

            return _()

        return wrapper

    class text:
        @staticmethod
        @only_once
        @as_is_decorator
        def complete(f: Complete | AsyncComplete):
            return patch._make_complete_wrapper(plant_text_completions, f)

        @staticmethod
        @only_once
        @as_is_decorator
        def generate(f: Generate | AsyncGenerate):
            return patch._make_generate_wrapper(plant_text_completions, f)

        @staticmethod
        @only_once
        @as_is_decorator
        def auto(f: Complete | AsyncComplete | Generate | AsyncGenerate):
            return patch._make_dispatch_wrapper(plant_text_completions, f)

        def __new__(cls, f: Complete | AsyncComplete | Generate | AsyncGenerate):
            return cls.auto(f)

        @staticmethod
        @only_once
        def llm(LLMClass: type[LLM]):
            class TraceableLLM(LLMClass):
                @property
                def complete(self):
                    return patch.text.complete(super().complete)

                @property
                def generate(self):
                    return patch.text.generate(super().generate)

            return TraceableLLM

        # for backward compatibility
        acomplete = complete
        agenerate = generate

    class chat:
        @staticmethod
        @only_once
        @as_is_decorator
        def complete(f: Complete | AsyncComplete):
            return patch._make_complete_wrapper(plant_chat_completions, f)

        @staticmethod
        @only_once
        def generate(f: Generate | AsyncGenerate):
            return patch._make_generate_wrapper(plant_chat_completions, f)

        @staticmethod
        @only_once
        def llm(LLMClass: type[LLM]):
            class TraceableLLM(LLMClass):
                @property
                def complete(self):
                    return patch.chat.complete(super().complete)

                @property
                def generate(self):
                    return patch.chat.generate(super().generate)

            return TraceableLLM

        @staticmethod
        @only_once
        @as_is_decorator
        def auto(f: Complete | AsyncComplete | Generate | AsyncGenerate):
            return patch._make_dispatch_wrapper(plant_chat_completions, f)

        T = TypeVar("T", bound=Complete | AsyncComplete | Generate | AsyncGenerate)

        def __new__(cls, f: T) -> T:
            return patch._make_dispatch_wrapper(plant_chat_completions, f)  # type: ignore

        # for backward compatibility
        acomplete = complete
        agenerate = generate

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
                parent_run = ensure_parent_run(None if context is None else find_runs(context)[-1])

                prompt = super().render(context, callbacks)

                parent_run.event(
                    name="render",
                    input={
                        "template": self.template.text,
                        "context": {} if context is None else ensure_serializable(clean(context)),
                    },
                    output=prompt,
                    start_time=utcnow(),
                )

                return prompt

            async def arender(self, context: Context | None = None, callbacks=None):
                parent_run = ensure_parent_run(None if context is None else find_runs(context)[-1])

                prompt = await super().arender(context, callbacks)

                parent_run.event(
                    name="arender",
                    input={
                        "template": self.template.text,
                        "context": {} if context is None else ensure_serializable(clean(context)),
                    },
                    output=prompt,
                    start_time=utcnow(),
                )

                return prompt

        return TraceableNode
