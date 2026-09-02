# Batch processing

`cheat_at_search.batch` provides `BatchProcessor`, a resumable task-oriented
wrapper around the OpenAI Batch API.

## Client

The processor accepts an optional `AsyncOpenAI` instance. If omitted, it creates
one using:

```python
key = key_for_provider("openai")
client = AsyncOpenAI(api_key=key)
```

Batch input files and state are stored below:

```python
batch_file_dir = ensure_data_subdir("batch_file_cache_dir")
```

The client must be mocked in unit tests.

## Tasks

The task is the central unit of work. A workload is an iterable of task
objects, and each task corresponds to one row in an OpenAI batch submission.

Tasks must provide:

- `id`: a unique identifier, serialized as OpenAI's `custom_id`.
- `to_json() -> dict`: the JSONL request for one batch row.
- `is_done() -> bool`: whether the task's external result already exists.
- `finish(output) -> bool`: an async method that consumes one output row and
  completes the task. `True` marks the task `done`; `False` marks it `failed`.

The base task class raises `NotImplementedError` for these methods. Users may
subclass it for image, text, or any other supported batch operation.

`ImageGenerationTask` and `TextGenerationTask` provide request serialization
examples. They do not implement application-specific `is_done` or `finish`
behavior.

## Processing

`BatchProcessor.process(tasks, batch_size=100)` materializes the iterable,
validates unique task IDs, and then:

1. Marks externally completed tasks as `done`.
2. Skips tasks already marked `done` or associated with an active submitted
   batch.
3. Retries tasks whose latest CSV status is `failed`, `expired`, or
   `cancelled` on a later invocation.
4. Serializes and submits pending tasks in batches, including the final partial
   batch.
5. Polls active batches asynchronously until each is terminal.
6. Matches output rows by `custom_id`, not output order.
7. Calls `finish` concurrently with bounded concurrency.

OpenAI API calls use `AsyncOpenAI`. A failure to submit a batch is logged and
does not create task rows; those tasks can therefore be retried on a later
invocation. A failed, expired, or cancelled OpenAI batch marks all of its tasks
as `failed`. If `finish` returns `False` or raises an exception, the task is
also marked `failed` and the error is logged.

Failures are not retried during the same invocation. They are eligible for
retry when the workload is run again.

## State

The task database is a CSV file at
`batch_file_cache_dir/batch_status.csv`, loaded as a pandas DataFrame for
convenience. Its schema is:

```csv
task_id,batch_id,status
```

There is no separate batch-to-task mapping. The CSV is the sole source of
truth. State writes use a temporary file followed by replacement to avoid
leaving a partially written CSV.

Task completion should be idempotent because a process may stop after an
external side effect succeeds but before its CSV status is persisted.

All tasks within one submitted batch must target the same OpenAI endpoint.
