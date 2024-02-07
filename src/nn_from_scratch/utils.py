from itertools import islice

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

progress = Progress(
    TextColumn("Training..."),
    BarColumn(),
    TaskProgressColumn(),
    MofNCompleteColumn(),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
)


def chunks(iterable, chunk_size):
    """Splits the iterable into chunks of size chunk_size."""
    while True:
        chunk = list(islice(iterable, chunk_size))
        if not chunk:
            return
        yield chunk
