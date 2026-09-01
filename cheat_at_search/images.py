from openai import OpenAI
from cheat_at_search.data_dir import key_for_provider, ensure_data_subdir, mount
import pandas as pd
from cheat_at_search.wands_data import corpus
import os
import time
import asyncio
import base64


mount(use_gdrive=False)


key = key_for_provider("openai")


class OpenAIImageGenerator:

    def __init__(self,
                 openai: OpenAI | None = None,
                 model: str = "gpt-image-1.5",
                 size: str = "1024x1024",
                 quality: str = "low"):
        if openai is None:
            openai = OpenAI(api_key=key)
        self.openai = openai
        self.model = model
        self.quality = quality
        self.size = "1024x1024"

    def _prompt(self, title: str, description: str) -> str:
        prompt = "You generate images of e-commerce products for display in search results."
        prompt += "Do not put the text in teh product itself. Just a nice image."
        prompt += f"Title: {title}\nDescription: {description}\nGenerate an image based on the above title and description."
        return prompt

    def _batch_request(self, title: str, description: str, doc_id: str) -> list[bytes]:
        """Batch body to OpenAI."""
        prompt = self._prompt(title, description)
        body = {
            "model": self.model,
            "prompt": prompt,
            "size": self.size,
            "quality": self.quality,
        }
        request = {
            "custom_id": str(doc_id),
            "method": "POST",
            "url": "/v1/images/generations",
            "body": body,
        }
        return request

    def batch_requests_to_file(self, products: pd.DataFrame,
                               start: int,
                               num_products: int,
                               file_path: str) -> None:
        """Generate batch requests for OpenAI image generation and write to a JSONL file."""
        products = products.iloc[start:start + num_products]
        requests = products[["title", "description", "doc_id"]].apply(
            lambda row: self._batch_request(row["title"], row["description"], row["doc_id"]),
            axis=1,
        )
        # write as jsonl
        requests.to_json(file_path, orient="records", lines=True)

    def submit_batch(self, batch_file_path: str) -> None:
        """Submit batch, block until done."""
        batch_file = self.openai.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch",
        )
        batch_file_id = batch_file.id

        print(f"Batch file {batch_file_id} created. Waiting for processed status...")

        while True:
            batch_file = self.openai.files.retrieve(batch_file_id)
            if batch_file.status == "processed":
                break
            print(f"Batch file {batch_file_id} status: {batch_file.status}. Waiting for processed status...")
            time.sleep(10)

        print("File ready, Submitting batch...")

        batch = self.openai.batches.create(
            input_file_id=batch_file_id,
            endpoint="/v1/images/generations",
            completion_window="24h",
        )
        # Loop until we confirm its in_progress
        while True:
            batch = self.openai.batches.retrieve(batch.id)
            if batch.status in ["in_progress", "completed", "failed"]:
                break
            print(f"Batch {batch.id} status: {batch.status}. Waiting for processing...")
            time.sleep(10)
        if batch.status == "failed":
            raise RuntimeError(f"Batch {batch.id} failed.")
        return batch.id

    async def await_branch(self, batch_id: str, output_file_path: str) -> None:
        print(f"Awaiting batch {batch_id} completion...")
        while True:
            batch = self.openai.batches.retrieve(batch_id)
            if batch.status in ["completed", "failed"]:
                break
            print(f"Batch {batch_id} status: {batch.status}. Waiting for completion...")
            await asyncio.sleep(60 * 5)  # Wait for 5 minutes before checking again
        if batch.status == "failed":
            raise RuntimeError(f"Batch {batch_id} failed.")
        elif batch.status == "completed":
            print(f"Batch {batch_id} completed successfully.")

        result = self.openai.files.content(batch.output_file_id)
        result.write_to_file(output_file_path)

    def process_output_batch(self, output_file_path: str, images_dir: str) -> pd.DataFrame:
        """Process the output batch file and save each with product id image."""
        df = pd.read_json(output_file_path, lines=True)
        resp = pd.json_normalize(df["response"])
        columns_to_copy = ["request_id", "status_code", "body.data", "body.size"]
        for col in columns_to_copy:
            df[col] = resp[col]
        df["doc_id"] = df["custom_id"]
        for idx, row in df.iterrows():
            image_data = row["body.data"][0]["b64_json"] if row["body.data"] else None
            if image_data:
                image_path = os.path.join(images_dir, f"{row['doc_id']}.png")
                with open(image_path, "wb") as f:
                    image_bytes = base64.b64decode(image_data)
                    f.write(image_bytes)

    def generate(self, title: str, description: str) -> bytes:
        prompt = self._prompt(title, description)
        result = self.openai.images.generate(
            model=self.model,
            prompt=prompt,
            size=self.size,
            quality="medium",
        )
        return result.data[0].b64_json

    def generate_to(self, title: str, description: str, file_path: str) -> None:
        image_data = self.generate(title, description)
        with open(file_path, "wb") as f:
            f.write(image_data)


def main(images_dir, batch_size=10, num_batches=1, force=False):
    """Write images to images_dir for every wands product."""
    # Example usage
    batch_file_dir = ensure_data_subdir("image_batch_files")

    # Submit all batches
    submitted_batches = []
    processable_batches = []

    def in_file(batch_idx):
        return os.path.join(batch_file_dir, f"wands_images_{batch_idx}_{batch_size}.jsonl")

    def out_file(batch_idx):
        return os.path.join(batch_file_dir, f"wands_images_output_{batch_idx}_{batch_size}.jsonl")

    generator = OpenAIImageGenerator(model="gpt-image-1")

    for batch_idx in range(0, len(corpus), batch_size):
        print(f"Processing batch {batch_idx}...")
        print(f"Check out file: {out_file(batch_idx)}")
        if not force and os.path.exists(out_file(batch_idx)):
            print(f"Batch {batch_idx} already processed, skipping.")
            processable_batches.append(out_file(batch_idx))
        else:
            print(f"Submitting batch {batch_idx}...")
            generator.batch_requests_to_file(corpus,
                                             start=batch_idx,
                                             num_products=batch_size,
                                             file_path=in_file(batch_idx))
            batch_id = generator.submit_batch(in_file(batch_idx))
            submitted_batches.append(batch_id)
        if (batch_idx + batch_size) >= (num_batches * batch_size):
            break

    # Await all pending batches we don't have files for yet
    def wait_for_batches():
        async def runner():
            processed_files = []
            tasks = {}
            for idx, batch_id in enumerate(submitted_batches):
                idx = idx * batch_size
                op_file = out_file(batch_idx)
                print(f"Output saved to {op_file}")
                task = asyncio.create_task(generator.await_branch(batch_id, op_file))
                tasks[task] = (batch_id, op_file)

            pending = set(tasks.keys())
            while pending:
                done, pending = await asyncio.wait(pending,
                                                   return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await task
                    processed_files.append(tasks[task][1])
            return processed_files
        return asyncio.run(runner())

    addl_processed = wait_for_batches()
    processable_batches.extend(addl_processed)

    for processed_path in processable_batches:
        generator.process_output_batch(processed_path, images_dir)
        print(f"{processed_path} images written to {images_dir}")


if __name__ == "__main__":
    images_dir = ensure_data_subdir("images")
    main(images_dir, batch_size=100, num_batches=10)
