"""Generate WANDS product images through the task-based batch processor."""

import argparse
import asyncio
import base64
from pathlib import Path

from cheat_at_search.batch import ImageGenerationTask
from cheat_at_search.batch import BatchProcessor
from cheat_at_search.data_dir import ensure_data_subdir, mount
from cheat_at_search.wands_data import corpus


class WandsImageTask(ImageGenerationTask):
    """Generate and persist one WANDS product image."""

    def __init__(
        self,
        doc_id: str,
        title: str,
        description: str,
        images_dir: str | Path,
        model: str = "gpt-image-1.5",
    ):
        prompt = (
            "You generate images of e-commerce products for display in search results. "
            "Do not put text in the product itself. Create a clean product image.\n"
            f"Title: {title}\n"
            f"Description: {description}\n"
            "Generate an image based on the title and description."
        )
        super().__init__(id=str(doc_id), prompt=prompt, model=model, quality="low")
        self.images_dir = Path(images_dir)

    @property
    def image_path(self) -> Path:
        return self.images_dir / f"{self.id}.png"

    def is_done(self) -> bool:
        return self.image_path.exists()

    async def finish(self, output: dict) -> bool:
        response = output.get("response", {})
        if response.get("status_code", 500) >= 400:
            return False
        data = response.get("body", {}).get("data", [])
        if not data or not data[0].get("b64_json"):
            return False

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.image_path.write_bytes(base64.b64decode(data[0]["b64_json"]))
        return self.is_done()


def make_tasks(
    images_dir: str | Path,
    start: int = 0,
    count: int = 30,
) -> list[WandsImageTask]:
    """Create tasks for a slice of the WANDS corpus."""
    products = corpus.iloc[start : start + count]
    return [
        WandsImageTask(
            doc_id=row["doc_id"],
            title=row["title"],
            description=row["description"],
            images_dir=images_dir,
        )
        for _, row in products.iterrows()
    ]


async def run(
    images_dir: str | Path | None = None,
    start: int = 0,
    num_batches: int = 3,
    batch_size: int = 10,
    poll_seconds: float = 60,
) -> None:
    """Submit a few WANDS batches and wait for every task to finish."""
    if images_dir is None:
        images_dir = ensure_data_subdir("wands_images")
    tasks = make_tasks(images_dir, start=start, count=num_batches * batch_size)
    processor = BatchProcessor()
    await processor.process(
        tasks,
        batch_size=batch_size,
        poll_seconds=poll_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    asyncio.run(
        run(
            images_dir=args.images_dir,
            start=args.start,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            poll_seconds=args.poll_seconds,
        )
    )


if __name__ == "__main__":
    mount(use_gdrive=False)
    main()
