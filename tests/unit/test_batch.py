import asyncio
import base64
import functools
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
from PIL import Image

from cheat_at_search.batch import BatchProcessor, BatchTask
from cheat_at_search.batch import ImageGenerationTask, TextGenerationTask


def run_async(test_fn):
    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(test_fn(*args, **kwargs))

    return wrapper


def mock_openai(batch_ids=("batch-1",)):
    client = MagicMock()
    client.files.create = AsyncMock(return_value=SimpleNamespace(id="file-1"))
    client.batches.create = AsyncMock(
        side_effect=[SimpleNamespace(id=batch_id) for batch_id in batch_ids]
    )
    client.batches.retrieve = AsyncMock()
    client.files.content = AsyncMock()
    return client


class RecordingTask(BatchTask):
    def __init__(self, task_id, finish_result=True, finish_error=None):
        self.id = str(task_id)
        self.finish_result = finish_result
        self.finish_error = finish_error
        self.finished = False

    def to_json(self):
        return {
            "custom_id": self.id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {"model": "test-model", "input": self.id},
        }

    def is_done(self):
        return False

    async def finish(self, output):
        self.finished = True
        if self.finish_error:
            raise self.finish_error
        return self.finish_result


def output_for(task_id):
    return {"custom_id": str(task_id), "response": {"status_code": 200}}


def configure_completed_batch(client, task_ids):
    client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="output-1"
    )
    client.files.content.return_value = SimpleNamespace(
        text="\n".join(json.dumps(output_for(task_id)) for task_id in task_ids)
    )


def test_generation_tasks_serialize_requests():
    image = ImageGenerationTask("doc-1", "a chair", quality="low")
    text = TextGenerationTask("query-1", "expand this", "gpt-test", system_prompt="Be brief")

    image_json = image.to_json()
    text_json = text.to_json()

    assert image_json["custom_id"] == "doc-1"
    assert image_json["url"] == "/v1/images/generations"
    assert image_json["body"]["quality"] == "low"
    assert text_json["custom_id"] == "query-1"
    assert text_json["url"] == "/v1/responses"
    assert text_json["body"]["input"][0] == {
        "role": "system",
        "content": "Be brief",
    }


@patch("cheat_at_search.batch.AsyncOpenAI")
@patch("cheat_at_search.batch.key_for_provider", return_value="test-key")
def test_processor_constructs_async_openai_client(key_for_provider, async_openai, tmp_path):
    BatchProcessor(cache_dir=tmp_path)

    key_for_provider.assert_called_once_with("openai")
    async_openai.assert_called_once_with(api_key="test-key")


def test_done_csv_row_is_reopened_when_task_is_not_done(tmp_path):
    status_path = tmp_path / "batch_status.csv"
    pd.DataFrame(
        [{"task_id": "a", "batch_id": "old-batch", "status": "done"}]
    ).to_csv(status_path, index=False)
    client = mock_openai()
    configure_completed_batch(client, ["a"])
    task = RecordingTask("a")

    result = asyncio.run(
        BatchProcessor(client, tmp_path).process([task], batch_size=1, poll_seconds=0)
    )

    client.batches.create.assert_awaited_once()
    assert result.iloc[0]["status"] == "done"


@run_async
async def test_process_submits_final_partial_batch_and_finishes_by_custom_id(tmp_path):
    client = mock_openai(batch_ids=("batch-1", "batch-2"))
    tasks = [RecordingTask("a"), RecordingTask("b"), RecordingTask("c")]
    configure_completed_batch(client, ["c", "a", "b"])

    result = await BatchProcessor(client, tmp_path).process(
        tasks, batch_size=2, poll_seconds=0
    )

    assert client.batches.create.await_count == 2
    assert [call.kwargs["endpoint"] for call in client.batches.create.await_args_list] == [
        "/v1/responses",
        "/v1/responses",
    ]
    assert result["status"].tolist() == ["done", "done", "done"]
    assert all(task.finished for task in tasks)
    saved = pd.read_csv(tmp_path / "batch_status.csv", dtype=str)
    assert saved[["task_id", "batch_id", "status"]].to_dict("records") == [
        {"task_id": "a", "batch_id": "batch-1", "status": "done"},
        {"task_id": "b", "batch_id": "batch-1", "status": "done"},
        {"task_id": "c", "batch_id": "batch-2", "status": "done"},
    ]


@run_async
async def test_process_marks_false_and_exception_finishes_failed(tmp_path, caplog):
    client = mock_openai()
    tasks = [
        RecordingTask("false", finish_result=False),
        RecordingTask("error", finish_error=RuntimeError("finish failed")),
        RecordingTask("success"),
    ]
    configure_completed_batch(client, [task.id for task in tasks])

    result = await BatchProcessor(client, tmp_path).process(tasks, batch_size=3, poll_seconds=0)

    assert result["status"].tolist() == ["failed", "failed", "done"]
    assert "Batch task error failed while finishing" in caplog.text


@run_async
async def test_failed_openai_batch_marks_all_tasks_failed(tmp_path, caplog):
    client = mock_openai()
    client.batches.retrieve.return_value = SimpleNamespace(status="expired")
    tasks = [RecordingTask("a"), RecordingTask("b")]

    result = await BatchProcessor(client, tmp_path).process(tasks, batch_size=2, poll_seconds=0)

    assert result["status"].tolist() == ["failed", "failed"]
    assert "ended with status expired" in caplog.text
    assert not any(task.finished for task in tasks)


@run_async
async def test_process_resumes_active_batch_without_resubmitting(tmp_path):
    status_path = tmp_path / "batch_status.csv"
    pd.DataFrame(
        [{"task_id": "a", "batch_id": "old-batch", "status": "submitted"}]
    ).to_csv(status_path, index=False)
    client = mock_openai()
    client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="output-1"
    )
    client.files.content.return_value = SimpleNamespace(
        text=json.dumps(output_for("a"))
    )
    task = RecordingTask("a")

    result = await BatchProcessor(client, tmp_path).process([task], poll_seconds=0)

    client.batches.create.assert_not_awaited()
    assert client.batches.retrieve.await_args.args == ("old-batch",)
    assert result.iloc[0]["status"] == "done"
    assert task.finished


@run_async
async def test_process_retries_failed_task_on_later_run(tmp_path):
    pd.DataFrame(
        [{"task_id": "a", "batch_id": "old-batch", "status": "failed"}]
    ).to_csv(tmp_path / "batch_status.csv", index=False)
    client = mock_openai()
    configure_completed_batch(client, ["a"])

    result = await BatchProcessor(client, tmp_path).process(
        [RecordingTask("a")], batch_size=1, poll_seconds=0
    )

    client.batches.create.assert_awaited_once()
    assert result.iloc[-1]["status"] == "done"


@run_async
async def test_missing_output_is_failed_and_logged(tmp_path, caplog):
    client = mock_openai()
    client.batches.retrieve.return_value = SimpleNamespace(
        status="completed", output_file_id="output-1"
    )
    client.files.content.return_value = SimpleNamespace(text=json.dumps(output_for("other")))
    task = RecordingTask("missing")

    result = await BatchProcessor(client, tmp_path).process([task], poll_seconds=0)

    assert result.iloc[0]["status"] == "failed"
    assert "No output returned for batch task missing" in caplog.text


def test_image_finish_uploads_valid_png_and_is_done_checks_bucket():
    from cheat_at_search.images2 import WandsImageTask

    image_buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(image_buffer, format="PNG")
    png_bytes = image_buffer.getvalue()

    class Blob:
        def __init__(self):
            self.data = None

        def exists(self):
            return self.data is not None

        def download_as_bytes(self):
            return self.data

        def upload_from_string(self, data, content_type):
            assert content_type == "image/png"
            self.data = data

    class Bucket:
        def __init__(self):
            self.blob_instance = Blob()

        def blob(self, name):
            assert name == "wands/images/doc-1.png"
            return self.blob_instance

    bucket = Bucket()
    task = WandsImageTask("doc-1", "Chair", "A chair", bucket)
    encoded = base64.b64encode(png_bytes).decode()
    output = {
        "response": {
            "status_code": 200,
            "body": {"data": [{"b64_json": encoded}]},
        }
    }

    assert asyncio.run(task.finish(output))
    assert task.is_done()
    assert bucket.blob_instance.data == png_bytes
