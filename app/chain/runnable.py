from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, SerializeAsAny

I = TypeVar("I")
O = TypeVar("O")
M = TypeVar("M")


class Runnable(BaseModel, Generic[I, O]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = None

    def invoke(self, data: I) -> O:
        raise NotImplementedError("Subclasses must implement invoke()")

    def __or__(self, other: Any) -> "RunnableSequence":
        if isinstance(other, Runnable):
            return RunnableSequence.model_construct(first=self, second=other)
        if callable(other):
            return RunnableSequence.model_construct(
                first=self,
                second=RunnableLambda.model_construct(func=other, name=other.__name__),
            )
        return NotImplemented

    def __ror__(self, other: Any) -> Any:
        if callable(other):
            return RunnableSequence.model_construct(
                first=RunnableLambda.model_construct(func=other),
                second=self,
                name=other.__name__,
            )
        return NotImplemented


class RunnableLambda(Runnable[I, O]):
    func: Callable[[I], O]

    def invoke(self, data: I) -> O:
        return self.func(data)


class RunnableSequence(Runnable[I, O], Generic[I, M, O]):
    first: SerializeAsAny[Runnable[I, M]]
    second: SerializeAsAny[Runnable[M, O]]

    def invoke(self, data: I) -> O:
        return self.second.invoke(self.first.invoke(data))
