from inspect import isasyncgen, isawaitable, isgenerator
from typing import TypeVar, cast

from langfuse import Langfuse, LangfuseSpan
from promplate.chain.node import BaseCallback, Chain, Interruptable, Node
from promplate.llm.base import LLM, AsyncComplete, AsyncGenerate, Complete, Generate
from promplate.prompt.template import Context

from .env import env
from .utils import as_is_decorator, cache, clean, diff_context, ensure_serializable, get_versions, name, only_once, split_model_parameters, wraps

MaybeRun = LangfuseSpan | None

LF_PARENT = "__lf_parent__"


def find_runs(*contexts: Context) -> list[MaybeRun]:
    for context in contexts:
        if run_stack := context.get(LF_PARENT):
            return run_stack

    return [None]


@cache
def get_client():
    return Langfuse(
        public_key=env.langfuse_public_key,
        secret_key=env.langfuse_secret_key,
        host=env.langfuse_host,
    )


class TraceCallback(BaseCallback):
    def on_enter(self, node, context: Context | None, config: Context):
        context_in = self.context_in = {} if context is None else clean(context)

        langfuse = get_client()
        runs = self.runs = find_runs(config, context_in)

        # Create a span using context manager pattern
        self.span_ctx = langfuse.start_as_current_span(name=str(node), input=ensure_serializable(context_in))
        run = self.span_ctx.__enter__()

        if context is None:
            context = {}

        context[LF_PARENT] = config[LF_PARENT] = [*runs, run]

        return context, config

    def on_leave(self, _, context: Context, config: Context):  # type: ignore
        context_out = clean(context)

        runs = find_runs(config, context)
        run = cast(LangfuseSpan, runs[-1])

        # Update span output
        run.update(output=ensure_serializable(diff_context(self.context_in, context_out)))

        # Exit the context manager
        self.span_ctx.__exit__(None, None, None)

        context[LF_PARENT] = config[LF_PARENT] = runs[:-1]

        return context, config


T = TypeVar("T", bound=Interruptable)


class patch:
    @staticmethod
    def _make_complete_wrapper(original):
        @wraps(original)
        def wrapper(prompt, /, **config):
            langfuse = get_client()
            config.pop(LF_PARENT, None)

            config_params, extras = split_model_parameters(config)
            model = str(config_params.pop("model", None))

            gen_ctx = langfuse.start_as_current_generation(
                name=name(original),
                input=prompt,
                model=model,
                model_parameters=config_params,
                metadata=(extras or {}) | get_versions("promplate", "promplate-trace", "langfuse"),
            )

            gen = gen_ctx.__enter__()

            try:
                out = original(prompt, **config)

                if isawaitable(out):

                    async def _():
                        try:
                            result = await out
                            gen.update(output=result)
                            return result
                        finally:
                            gen_ctx.__exit__(None, None, None)

                    return _()

                gen.update(output=out)
                gen_ctx.__exit__(None, None, None)
                return out
            except Exception as e:
                gen_ctx.__exit__(type(e), e, e.__traceback__)
                raise

        return wrapper

    @staticmethod
    def _make_generate_wrapper(original):
        @wraps(original)
        def wrapper(prompt, /, **config):
            langfuse = get_client()
            config.pop(LF_PARENT, None)

            config_params, extras = split_model_parameters(config)
            model = str(config_params.pop("model", None))

            gen_ctx = langfuse.start_as_current_generation(
                name=name(original),
                input=prompt,
                model=model,
                model_parameters=config_params,
                metadata=(extras or {}) | get_versions("promplate", "promplate-trace", "langfuse"),
            )

            gen = gen_ctx.__enter__()
            stream = original(prompt, **config)

            if isasyncgen(stream):

                async def _():
                    out = ""
                    try:
                        async for delta in stream:
                            out += delta
                            yield delta
                        gen.update(output=out)
                    finally:
                        gen_ctx.__exit__(None, None, None)

                return _()

            assert isgenerator(stream)

            def _():
                out = ""
                try:
                    for delta in stream:
                        out += delta
                        yield delta
                    gen.update(output=out)
                finally:
                    gen_ctx.__exit__(None, None, None)

            return _()

        return wrapper

    @staticmethod
    def _make_auto_wrapper(original):
        @wraps(original)
        def wrapper(prompt, /, **config):
            langfuse = get_client()
            config.pop(LF_PARENT, None)

            config_params, extras = split_model_parameters(config)
            model = str(config_params.pop("model", None))

            gen_ctx = langfuse.start_as_current_generation(
                name=name(original),
                input=prompt,
                model=model,
                model_parameters=config_params,
                metadata=(extras or {}) | get_versions("promplate", "promplate-trace", "langfuse"),
            )

            gen = gen_ctx.__enter__()
            res = original(prompt, **config)

            if isasyncgen(res):

                async def _():
                    out = ""
                    try:
                        async for delta in res:
                            out += delta
                            yield delta
                        gen.update(output=out)
                    finally:
                        gen_ctx.__exit__(None, None, None)

                return _()

            if isawaitable(res):

                async def _():
                    try:
                        result = await res
                        gen.update(output=result)
                        return result
                    finally:
                        gen_ctx.__exit__(None, None, None)

                return _()

            if isinstance(res, str):
                try:
                    gen.update(output=res)
                    return res
                finally:
                    gen_ctx.__exit__(None, None, None)

            assert isgenerator(res)

            def _():
                out = ""
                try:
                    for delta in res:
                        out += delta
                        yield delta
                    gen.update(output=out)
                finally:
                    gen_ctx.__exit__(None, None, None)

            return _()

        return wrapper

    class text:
        @staticmethod
        @only_once
        @as_is_decorator
        def complete(f: Complete | AsyncComplete):
            return patch._make_complete_wrapper(f)

        @staticmethod
        @only_once
        @as_is_decorator
        def generate(f: Generate | AsyncGenerate):
            return patch._make_generate_wrapper(f)

        @staticmethod
        @only_once
        @as_is_decorator
        def auto(f: Complete | AsyncComplete | Generate | AsyncGenerate):
            return patch._make_auto_wrapper(f)

        T = TypeVar("T", bound=Complete | AsyncComplete | Generate | AsyncGenerate)

        def __new__(cls, f: T) -> T:
            return cls.auto(f)

        @staticmethod
        @only_once
        def llm(LLMClass: type[LLM]):
            class TraceableLLM(LLMClass):
                @property
                def complete(self):  # type: ignore
                    return patch.text.complete(super().complete)

                @property
                def generate(self):  # type: ignore
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
            return patch._make_complete_wrapper(f)

        @staticmethod
        @only_once
        @as_is_decorator
        def generate(f: Generate | AsyncGenerate):
            return patch._make_generate_wrapper(f)

        @staticmethod
        @only_once
        @as_is_decorator
        def auto(f: Complete | AsyncComplete | Generate | AsyncGenerate):
            return patch._make_auto_wrapper(f)

        T = TypeVar("T", bound=Complete | AsyncComplete | Generate | AsyncGenerate)

        def __new__(cls, f: T) -> T:
            return cls.auto(f)

        @staticmethod
        @only_once
        def llm(LLMClass: type[LLM]):
            class TraceableLLM(LLMClass):
                @property
                def complete(self):  # type: ignore
                    return patch.chat.complete(super().complete)

                @property
                def generate(self):  # type: ignore
                    return patch.chat.generate(super().generate)

            return TraceableLLM

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
                langfuse = get_client()

                prompt = super().render(context, callbacks)

                # Create an event to log the render operation
                with langfuse.start_as_current_span(name="render") as span:
                    span.update(
                        input={
                            "template": self.template.text,
                            "context": {} if context is None else ensure_serializable(clean(context)),
                        },
                        output=prompt,
                    )

                return prompt

            async def arender(self, context: Context | None = None, callbacks=None):
                langfuse = get_client()

                prompt = await super().arender(context, callbacks)

                # Create an event to log the arender operation
                with langfuse.start_as_current_span(name="arender") as span:
                    span.update(
                        input={
                            "template": self.template.text,
                            "context": {} if context is None else ensure_serializable(clean(context)),
                        },
                        output=prompt,
                    )

                return prompt

        return TraceableNode
