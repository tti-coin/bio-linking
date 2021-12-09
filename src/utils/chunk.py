# ValueType = typing.TypeVar("ValueType")
# def chunk(iterable: collections.abc.Iterable[ValueType], chunk_size:int) -> Generator[List[ValueType]]:
def chunking(iterable, chunk_size:int):
    assert chunk_size > 0
    chunk = list()
    for val in iterable:
        chunk.append(val)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = list()
    if len(chunk) > 0:
        yield chunk
