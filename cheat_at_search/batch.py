"""Task-oriented, resumable processing for the OpenAI Batch API."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from openai import AsyncOpenAI

from cheat_at_search.data_dir import ensure_data_subdir, key_for_provider


logger = logging.getLogger(__name__)
_STATUS_COLUMNS = ["task_id", "batch_id", "status"]
_RETRYABLE_STATUSES = {"failed", "expired", "cancelled"}
_ACTIVE_STATUSES = {"submitted", "in_progress"}


class BatchTask:
    """Base protocol for one unit of batch work."""

    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError

    def is_done(self) -> bool:
        raise NotImplementedError

    async def finish(self, output: dict[str, Any]) -> bool:
        raise NotImplementedError


@dataclass
class ImageGenerationTask(BatchTask):
    """An image-generation request in OpenAI batch JSONL format."""

    id: str
    prompt: str
    model: str = "gpt-image-1"
    size: str = "1024x1024"
    quality: str = "auto"
    extra_body: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "custom_id": str(self.id),
            "method": "POST",
            "url": "/v1/images/generations",
            "body": {
                "model": self.model,
                "prompt": self.prompt,
                "size": self.size,
                "quality": self.quality,
                **self.extra_body,
            },
        }


@dataclass
class TextGenerationTask(BatchTask):
    """A Responses API text-generation request in batch JSONL format."""

    id: str
    prompt: str
    model: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        input_value: str | list[dict[str, str]] = self.prompt
        if self.system_prompt is not None:
            input_value = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.prompt},
            ]
        body: dict[str, Any] = {
            "model": self.model,
            "input": input_value,
            **self.extra_body,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            body["max_output_tokens"] = self.max_output_tokens
        return {
            "custom_id": str(self.id),
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }


class BatchProcessor:
    """Submit and finish tasks while persisting task state in a CSV ledger."""

    def __init__(
        self,
        openai: AsyncOpenAI | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.openai = (
            openai
            if openai is not None
            else AsyncOpenAI(api_key=key_for_provider("openai"))
        )
        self.batch_file_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else ensure_data_subdir("batch_file_cache_dir")
        )
        self.batch_file_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.batch_file_dir / "batch_status.csv"
        if not self.status_path.exists():
            self._save_db(self._empty_db())

    @staticmethod
    def _empty_db() -> pd.DataFrame:
        return pd.DataFrame(columns=_STATUS_COLUMNS)

    def _load_db(self) -> pd.DataFrame:
        if not self.status_path.exists() or self.status_path.stat().st_size == 0:
            return self._empty_db()
        db = pd.read_csv(self.status_path, dtype={column: str for column in _STATUS_COLUMNS})
        missing = set(_STATUS_COLUMNS) - set(db.columns)
        if missing:
            raise ValueError(f"Batch status CSV is missing columns: {sorted(missing)}")
        return db[_STATUS_COLUMNS]

    def _save_db(self, db: pd.DataFrame) -> None:
        temporary_path = self.status_path.with_suffix(".csv.tmp")
        db[_STATUS_COLUMNS].to_csv(temporary_path, index=False)
        temporary_path.replace(self.status_path)

    def _input_path(self, batch_number: int) -> Path:
        return self.batch_file_dir / f"batch_{batch_number}.jsonl"

    def _append_tasks(self, db: pd.DataFrame, tasks: list[BatchTask], batch_id: str) -> pd.DataFrame:
        rows = pd.DataFrame(
            {
                "task_id": [str(task.id) for task in tasks],
                "batch_id": batch_id,
                "status": "submitted",
            }
        )
        return pd.concat([db, rows], ignore_index=True)

    async def _submit(self, tasks: list[BatchTask], batch_number: int) -> str:
        print(f"Preparing batch {batch_number} with {len(tasks)} tasks...")
        requests = [task.to_json() for task in tasks]
        endpoints = {request["url"] for request in requests}
        if len(endpoints) != 1:
            raise ValueError("All tasks in a batch must use the same OpenAI endpoint.")
        input_path = self._input_path(batch_number)
        with input_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(request) + "\n")
        with input_path.open("rb") as input_file:
            print(f"Uploading batch {batch_number} input file: {input_path}")
            uploaded = await self.openai.files.create(file=input_file, purpose="batch")
        batch = await self.openai.batches.create(
            input_file_id=uploaded.id,
            endpoint=next(iter(endpoints)),
            completion_window="24h",
        )
        print(f"Submitted batch {batch.id} with {len(tasks)} tasks.")
        return batch.id

    async def _fetch_content(self, batch: Any) -> list[dict[str, Any]]:
        content = await self.openai.files.content(batch.output_file_id)
        text = content.text if hasattr(content, "text") else str(content)
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    async def _finish_one(
        self,
        task: BatchTask,
        output: dict[str, Any] | None,
        semaphore: asyncio.Semaphore,
    ) -> bool:
        async with semaphore:
            if output is None:
                logger.error("No output returned for batch task %s", task.id)
                return False
            try:
                return bool(await task.finish(output))
            except Exception:
                logger.exception("Batch task %s failed while finishing", task.id)
                return False

    async def _process_batch(
        self,
        batch_id: str,
        tasks_by_id: dict[str, BatchTask],
        db: pd.DataFrame,
        finish_semaphore: asyncio.Semaphore,
    ) -> pd.DataFrame:
        print(f"Checking batch {batch_id}...")
        batch = await self.openai.batches.retrieve(batch_id)
        status = batch.status
        task_ids = db.loc[db["batch_id"] == batch_id, "task_id"].tolist()
        print(f"Batch {batch_id} status: {status} ({len(task_ids)} tasks)")
        if status in {"failed", "expired", "cancelled"}:
            logger.error("OpenAI batch %s ended with status %s", batch_id, status)
            db.loc[db["batch_id"] == batch_id, "status"] = "failed"
            return db
        if status != "completed":
            return db

        print(f"Downloading output for batch {batch_id}...")
        outputs = {str(output.get("custom_id")): output for output in await self._fetch_content(batch)}
        results = await asyncio.gather(
            *(
                self._finish_one(tasks_by_id[task_id], outputs.get(task_id), finish_semaphore)
                for task_id in task_ids
            )
        )
        for task_id, success in zip(task_ids, results):
            print(f"Task {task_id}: {'done' if success else 'failed'}")
            db.loc[db["task_id"] == task_id, "status"] = "done" if success else "failed"
        return db

    async def process(
        self,
        tasks: Iterable[BatchTask],
        batch_size: int = 100,
        poll_seconds: float = 60,
        finish_concurrency: int = 20,
    ) -> pd.DataFrame:
        """Process the complete workload, resuming active and retryable tasks."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        task_list = list(tasks)
        print(f"Loaded {len(task_list)} tasks.")
        task_by_id = {str(task.id): task for task in task_list}
        if len(task_by_id) != len(task_list):
            raise ValueError("Task IDs must be unique.")

        db = self._load_db()
        done_by_id = {str(task.id): task.is_done() for task in task_list}
        for task in task_list:
            task_id = str(task.id)
            if done_by_id[task_id]:
                existing = db["task_id"] == task_id
                db.loc[existing, "status"] = "done"

        pending: list[BatchTask] = []
        for task in task_list:
            rows = db[db["task_id"] == str(task.id)]
            status = rows.iloc[-1]["status"] if not rows.empty else None
            if done_by_id[str(task.id)] or status in _ACTIVE_STATUSES or status == "done":
                continue
            pending.append(task)

        batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
        print(f"{len(pending)} tasks pending across {len(batches)} batches.")
        submitted = await asyncio.gather(
            *(self._submit(batch, index) for index, batch in enumerate(batches)),
            return_exceptions=True,
        )
        for batch, batch_id in zip(batches, submitted):
            if isinstance(batch_id, Exception):
                logger.error("Could not submit batch: %s", batch_id)
                print(f"Batch submission failed: {batch_id}")
                continue
            db = self._append_tasks(db, batch, str(batch_id))
        self._save_db(db)

        finish_semaphore = asyncio.Semaphore(finish_concurrency)
        while True:
            active_batch_ids = db.loc[db["status"].isin(_ACTIVE_STATUSES), "batch_id"].unique()
            if not len(active_batch_ids):
                break
            batch_results = await asyncio.gather(
                *(
                    self._process_batch(
                        batch_id, task_by_id, db.copy(), finish_semaphore
                    )
                    for batch_id in active_batch_ids
                )
            )
            for batch_result in batch_results:
                batch_ids = batch_result["batch_id"].isin(active_batch_ids)
                db.loc[batch_ids, "status"] = batch_result.loc[batch_ids, "status"]
            self._save_db(db)
            if db["status"].isin(_ACTIVE_STATUSES).any():
                print(f"Waiting {poll_seconds} seconds before checking again...")
                await asyncio.sleep(poll_seconds)
        print("All batches finished.")
        return db
