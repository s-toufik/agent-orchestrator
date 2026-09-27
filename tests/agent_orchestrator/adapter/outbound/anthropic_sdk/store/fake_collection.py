from typing import Any


class FakeCursor:
    def __init__(self, documents: list[dict[str, Any]], projection: dict[str, int] | None) -> None:
        self._documents = documents
        self._projection = projection

    def sort(self, keys: list[tuple[str, int]]) -> FakeCursor:
        for field, direction in reversed(keys):
            self._documents.sort(key=lambda document: document[field], reverse=direction < 0)
        return self

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for document in self._documents:
            if self._projection:
                yield {k: v for k, v in document.items() if self._projection.get(k)}
            else:
                yield dict(document)


class FakeCollection:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.indexes: list[tuple[Any, dict[str, Any]]] = []

    async def create_index(self, keys: Any, **options: Any) -> str:
        self.indexes.append((keys, options))
        return "index"

    async def insert_many(self, documents: list[dict[str, Any]]) -> None:
        self.documents.extend(dict(document) for document in documents)

    async def update_many(self, query: dict[str, Any], update: dict[str, Any]) -> None:
        for document in self._matching(query):
            document.update(update["$set"])

    def find(self, query: dict[str, Any], projection: dict[str, int] | None = None) -> FakeCursor:
        return FakeCursor(self._matching(query), projection)

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        found = self._matching(query)
        return dict(found[0]) if found else None

    async def replace_one(
        self, query: dict[str, Any], replacement: dict[str, Any], upsert: bool = False
    ) -> None:
        self.documents = [d for d in self.documents if d not in self._matching(query)]
        self.documents.append({**query, **replacement})

    def _matching(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return [d for d in self.documents if all(d.get(k) == v for k, v in query.items())]
