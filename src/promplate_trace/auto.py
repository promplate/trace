from promplate.chain.node import Chain, Node
from promplate.llm.base import AsyncComplete, AsyncGenerate, Complete, Generate

from .env import env


class patch:
    class text:
        @staticmethod
        def complete(f: Complete):
            if env.langsmith:
                from . import langsmith

                f = langsmith.patch.text.complete(f)

            if env.langfuse:
                from . import langfuse

                f = langfuse.patch.text.complete(f)

            return f

        @staticmethod
        def generate(f: Generate):
            if env.langsmith:
                from . import langsmith

                f = langsmith.patch.text.generate(f)

            if env.langfuse:
                from . import langfuse

                f = langfuse.patch.text.generate(f)

            return f

        @staticmethod
        def acomplete(f: AsyncComplete):
            if env.langsmith:
                from . import langsmith

                f = langsmith.patch.text.acomplete(f)

            if env.langfuse:
                from . import langfuse

                f = langfuse.patch.text.acomplete(f)

            return f

        @staticmethod
        def agenerate(f: AsyncGenerate):
            if env.langsmith:
                from . import langsmith

                f = langsmith.patch.text.agenerate(f)

            if env.langfuse:
                from . import langfuse

                f = langfuse.patch.text.agenerate(f)

            return f

    class chat:
        @staticmethod
        def complete(f: Complete):
            if env.langsmith:
                from . import langsmith

                f = langsmith.patch.chat.complete(f)
            if env.langfuse:
                from . import langfuse

                f = langfuse.patch.chat.complete(f)

            return f

        @staticmethod
        def generate(f: Generate):
            if env.langsmith:
                from . import langsmith

                f = langsmith.patch.chat.generate(f)

            if env.langfuse:
                from . import langfuse

                f = langfuse.patch.chat.generate(f)

            return f

        @staticmethod
        def acomplete(f: AsyncComplete):
            if env.langsmith:
                from . import langsmith

                f = langsmith.patch.chat.acomplete(f)

            if env.langfuse:
                from . import langfuse

                f = langfuse.patch.chat.acomplete(f)

            return f

        @staticmethod
        def agenerate(f: AsyncGenerate):
            if env.langsmith:
                from . import langsmith

                f = langsmith.patch.chat.agenerate(f)

            if env.langfuse:
                from . import langfuse

                f = langfuse.patch.chat.agenerate(f)

            return f

    @staticmethod
    def chain(ChainClass: type[Chain]):
        if env.langsmith:
            from . import langsmith

            ChainClass = langsmith.patch.chain(ChainClass)

        if env.langfuse:
            from . import langfuse

            ChainClass = langfuse.patch.chain(ChainClass)

        return ChainClass

    @staticmethod
    def node(NodeClass: type[Node]):
        if env.langsmith:
            from . import langsmith

            NodeClass = langsmith.patch.node(NodeClass)

        if env.langfuse:
            from . import langfuse

            NodeClass = langfuse.patch.node(NodeClass)

        return NodeClass
