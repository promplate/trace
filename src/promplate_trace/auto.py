from traceback import print_exc

from promplate.chain.node import Chain, Node
from promplate.llm.base import AsyncComplete, AsyncGenerate, Complete, Generate

from .env import env


class patch:
    class text:
        @staticmethod
        def complete(f: Complete):
            if env.langsmith:
                try:
                    from . import langsmith

                    f = langsmith.patch.text.complete(f)
                except Exception:
                    print_exc()

            if env.langfuse:
                try:
                    from . import langfuse

                    f = langfuse.patch.text.complete(f)
                except Exception:
                    print_exc()

            return f

        @staticmethod
        def generate(f: Generate):
            if env.langsmith:
                try:
                    from . import langsmith

                    f = langsmith.patch.text.generate(f)
                except Exception:
                    print_exc()

            if env.langfuse:
                try:
                    from . import langfuse

                    f = langfuse.patch.text.generate(f)
                except Exception:
                    print_exc()

            return f

        @staticmethod
        def acomplete(f: AsyncComplete):
            if env.langsmith:
                try:
                    from . import langsmith

                    f = langsmith.patch.text.acomplete(f)
                except Exception:
                    print_exc()

            if env.langfuse:
                try:
                    from . import langfuse

                    f = langfuse.patch.text.acomplete(f)
                except Exception:
                    print_exc()

            return f

        @staticmethod
        def agenerate(f: AsyncGenerate):
            if env.langsmith:
                try:
                    from . import langsmith

                    f = langsmith.patch.text.agenerate(f)
                except Exception:
                    print_exc()

            if env.langfuse:
                try:
                    from . import langfuse

                    f = langfuse.patch.text.agenerate(f)
                except Exception:
                    print_exc()

            return f

    class chat:
        @staticmethod
        def complete(f: Complete):
            if env.langsmith:
                try:
                    from . import langsmith

                    f = langsmith.patch.chat.complete(f)
                except Exception:
                    print_exc()
            if env.langfuse:
                try:
                    from . import langfuse

                    f = langfuse.patch.chat.complete(f)
                except Exception:
                    print_exc()

            return f

        @staticmethod
        def generate(f: Generate):
            if env.langsmith:
                try:
                    from . import langsmith

                    f = langsmith.patch.chat.generate(f)
                except Exception:
                    print_exc()

            if env.langfuse:
                try:
                    from . import langfuse

                    f = langfuse.patch.chat.generate(f)
                except Exception:
                    print_exc()

            return f

        @staticmethod
        def acomplete(f: AsyncComplete):
            if env.langsmith:
                try:
                    from . import langsmith

                    f = langsmith.patch.chat.acomplete(f)
                except Exception:
                    print_exc()

            if env.langfuse:
                try:
                    from . import langfuse

                    f = langfuse.patch.chat.acomplete(f)
                except Exception:
                    print_exc()

            return f

        @staticmethod
        def agenerate(f: AsyncGenerate):
            if env.langsmith:
                try:
                    from . import langsmith

                    f = langsmith.patch.chat.agenerate(f)
                except Exception:
                    print_exc()

            if env.langfuse:
                try:
                    from . import langfuse

                    f = langfuse.patch.chat.agenerate(f)
                except Exception:
                    print_exc()

            return f

    @staticmethod
    def chain(ChainClass: type[Chain]):
        if env.langsmith:
            try:
                from . import langsmith

                ChainClass = langsmith.patch.chain(ChainClass)
            except Exception:
                print_exc()

        if env.langfuse:
            try:
                from . import langfuse

                ChainClass = langfuse.patch.chain(ChainClass)
            except Exception:
                print_exc()

        return ChainClass

    @staticmethod
    def node(NodeClass: type[Node]):
        if env.langsmith:
            try:
                from . import langsmith

                NodeClass = langsmith.patch.node(NodeClass)
            except Exception:
                print_exc()

        if env.langfuse:
            try:
                from . import langfuse

                NodeClass = langfuse.patch.node(NodeClass)
            except Exception:
                print_exc()

        return NodeClass
