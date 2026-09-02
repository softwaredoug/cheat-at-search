"""Generate WANDS product images through the task-based batch processor."""

import argparse
import asyncio
import base64
from io import BytesIO
import os

from cheat_at_search.batch import ImageGenerationTask
from cheat_at_search.batch import BatchProcessor
from cheat_at_search.data_dir import mount
from cheat_at_search.wands_data import corpus


class WandsImageTask(ImageGenerationTask):
    """Generate and upload one WANDS product image to GCS."""

    def __init__(
        self,
        doc_id: str,
        title: str,
        description: str,
        bucket,
        model: str = "gpt-image-1",
        verify_png=False,
    ):
        prompt = (
            "You generate images of e-commerce products for display in search results. "
            "Do not put text in the product itself. Create a clean product image.\n"
            f"Title: {title}\n"
            f"Description: {description}\n"
            "Generate an image based on the title and description."
        )
        super().__init__(id=str(doc_id), prompt=prompt, model=model, quality="low")
        self.bucket = bucket
        self.verify_png = verify_png

    @property
    def blob_name(self) -> str:
        return f"wands/images/{self.id}.png"

    def _blob(self):
        return self.bucket.blob(self.blob_name)

    def _uri(self) -> str:
        return f"gs://{getattr(self.bucket, 'name', '<bucket>')}/{self.blob_name}"

    def is_done(self) -> bool:
        blob = self._blob()
        print(f"Checking GCS image for task {self.id}: {self._uri()}")
        if not blob.exists():
            print(f"Task {self.id}: image does not exist")
            return False
        try:
            if self.verify_png:
                from PIL import Image

                with Image.open(BytesIO(blob.download_as_bytes())) as image:
                    image.verify()
                    is_png = image.format == "PNG"
                    print(f"Task {self.id}: image exists and PNG validation is {is_png}")
                    return is_png
            else:
                print(f"Task {self.id}: image exists, skipping PNG validation")
                return True
        except Exception:
            print(f"Task {self.id}: image exists but PNG validation failed")
            return False

    async def finish(self, output: dict) -> bool:
        print(f"Finishing image task {self.id}")
        response = output.get("response", {})
        if response.get("status_code", 500) >= 400:
            print(f"Task {self.id}: OpenAI response failed with status {response.get('status_code')}")
            return False
        data = response.get("body", {}).get("data", [])
        if not data or not data[0].get("b64_json"):
            print(f"Task {self.id}: OpenAI response did not contain image data")
            return False

        image_bytes = base64.b64decode(data[0]["b64_json"])
        print(f"Task {self.id}: uploading image to {self._uri()}")
        await asyncio.to_thread(
            self._blob().upload_from_string,
            image_bytes,
            content_type="image/png",
        )
        success = self.is_done()
        print(f"Task {self.id}: finish {'succeeded' if success else 'failed'}")
        return success


def gcs_bucket():
    """Return the WANDS image bucket used by the original image workflow."""
    from google.cloud import storage

    project = os.environ["GCLOUD_TRAINING_PROJECT"]
    return storage.Client(project=project).bucket("product-ai-images")


def make_tasks(
    bucket,
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
            bucket=bucket,
        )
        for _, row in products.iterrows()
    ]


async def run(
    bucket=None,
    start: int = 0,
    num_batches: int = 3,
    batch_size: int = 10,
    poll_seconds: float = 60,
) -> None:
    """Submit a few WANDS batches and wait for every task to finish."""
    if bucket is None:
        bucket = gcs_bucket()
    tasks = make_tasks(bucket, start=start, count=num_batches * batch_size)
    processor = BatchProcessor()
    await processor.process(
        tasks,
        batch_size=batch_size,
        poll_seconds=poll_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    asyncio.run(
        run(
            start=args.start,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            poll_seconds=args.poll_seconds,
        )
    )


if __name__ == "__main__":
    mount(use_gdrive=False)
    main()
