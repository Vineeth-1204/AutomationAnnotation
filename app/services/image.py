import asyncio
import io
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.concurrency import run_in_threadpool

import app.db.session as db_session
from app.models.image import Image
from app.models.processing_job import ProcessingJob
from app.repositories.image import ImageRepository
from app.repositories.processing_job import ProcessingJobRepository
from app.services.base import BaseService
from app.services.pipeline import ImagePipelineService
from app.utils.storage import StorageClient
from app.core.config import settings

logger = logging.getLogger(__name__)

class ImageService(BaseService[Image]):
    def __init__(self, repository: ImageRepository):
        super().__init__(repository)
        self.repository = repository
        self.storage_client = StorageClient()
        self.pipeline_service = ImagePipelineService()

    async def start_collection_job(
        self,
        *,
        db: AsyncSession,
        dataset_id: int,
        queries: List[str],
        limit_per_query: int = 5,
        creator_id: Optional[int] = None
    ) -> ProcessingJob:
        job_repo = ProcessingJobRepository(db)
        job_data = {
            "job_type": "image_collection",
            "status": "PENDING",
            "parameters": {
                "queries": queries,
                "limit_per_query": limit_per_query
            },
            "dataset_id": dataset_id,
            "creator_id": creator_id
        }
        job = await job_repo.create(job_data)
        return job

    async def run_collection_background_task(
        self,
        job_id: int,
        dataset_id: int,
        queries: List[str],
        limit_per_query: int
    ) -> None:
        logger.info(f"Starting background image collection job {job_id} for dataset {dataset_id}")
        
        async with db_session.SessionLocal() as db:
            job_repo = ProcessingJobRepository(db)
            job = await job_repo.get(job_id)
            if not job:
                logger.error(f"Job {job_id} not found in background task!")
                return
            
            job = await job_repo.update(job, {
                "status": "RUNNING",
                "started_at": datetime.utcnow()
            })
            await db.commit()

        # Load existing hashes to prevent duplicates across collection runs
        existing_hashes = set()
        async with db_session.SessionLocal() as db:
            res = await db.execute(
                select(Image.quality_metrics).where(Image.dataset_id == dataset_id)
            )
            existing_metrics = res.scalars().all()
            existing_hashes = {m.get("dhash") for m in existing_metrics if m and m.get("dhash")}

        url_query_pairs = self._get_image_urls_by_query(queries, limit_per_query)
        
        semaphore = asyncio.Semaphore(5)
        downloaded_images = []
        failed_count = 0
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            tasks = [
                self._download_and_process_single(
                    client=client,
                    url=url,
                    query=query,
                    semaphore=semaphore,
                    dataset_id=dataset_id,
                    existing_hashes=existing_hashes
                )
                for url, query in url_query_pairs
            ]
            
            results = await asyncio.gather(*tasks)
            
            for result in results:
                if result:
                    downloaded_images.append(result)
                else:
                    failed_count += 1

        async with db_session.SessionLocal() as db:
            image_repo = ImageRepository(db)
            for img_meta in downloaded_images:
                await image_repo.create(img_meta)

            job_repo = ProcessingJobRepository(db)
            job = await job_repo.get(job_id)
            
            result_stats = {
                "total_urls_found": len(url_query_pairs),
                "downloaded_count": len(downloaded_images),
                "failed_count": failed_count,
            }
            
            await job_repo.update(job, {
                "status": "COMPLETED",
                "completed_at": datetime.utcnow(),
                "result": result_stats
            })
            await db.commit()
            
        logger.info(f"Background image collection job {job_id} completed successfully")

    def _get_image_urls_by_query(self, queries: List[str], limit: int) -> List[tuple]:
        pairs = []
        for q in queries:
            base_offset = (abs(hash(q)) % 50) + 1
            for offset in range(limit):
                poke_id = base_offset + offset
                pairs.append((f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{poke_id}.png", q))
        return list(set(pairs))

    async def _download_and_process_single(
        self,
        client: httpx.AsyncClient,
        url: str,
        query: str,
        semaphore: asyncio.Semaphore,
        dataset_id: int,
        existing_hashes: set
    ) -> Optional[Dict[str, Any]]:
        content = await self._download_with_retry(client, url, semaphore)
        if not content:
            return None

        # Verify image format and detect corruption
        try:
            img = PILImage.open(io.BytesIO(content))
            img.verify()
            img = PILImage.open(io.BytesIO(content))
        except Exception as e:
            logger.warning(f"Rejected corrupted image from {url}: {str(e)}")
            return None

        # Quality Metric 1: Blur Rejection
        blur_score = await self.pipeline_service.calculate_blur_score(img)
        if blur_score < settings.IMAGE_BLUR_THRESHOLD:
            logger.warning(f"Rejected blurry image from {url} (score: {blur_score:.2f})")
            return None

        # Quality Metric 2: Perceptual Duplicate Hashing Check
        dhash = await self.pipeline_service.calculate_dhash(img)
        for h in existing_hashes:
            if self.pipeline_service.hamming_distance(dhash, h) < 4:
                logger.warning(f"Rejected duplicate image from {url} (dhash: {dhash})")
                return None
        existing_hashes.add(dhash)

        # Quality Metric 3: Semantic Similarity Filtering (CLIP or SigLIP)
        similarity_score = await self.pipeline_service.calculate_clip_similarity(query, img)
        if similarity_score < settings.IMAGE_SIMILARITY_THRESHOLD:
            logger.warning(f"Rejected low similarity image from {url} (similarity: {similarity_score:.4f} against '{query}')")
            return None

        # Normalization (format & resolution)
        target_res = settings.IMAGE_NORMALIZED_RESOLUTION
        normalized_bytes = await self.pipeline_service.normalize_image(
            img, target_size=(target_res, target_res)
        )
        
        # Save to storage (returns path/URL)
        filename = f"{uuid.uuid4()}.jpg"
        file_path = await self.storage_client.upload_image(filename, normalized_bytes)
        
        return {
            "filename": filename,
            "file_path": file_path,
            "width": target_res,
            "height": target_res,
            "dataset_id": dataset_id,
            "quality_metrics": {
                "dhash": dhash,
                "blur_score": blur_score,
                "similarity_score": similarity_score,
                "format": "jpeg",
                "resolution": f"{target_res}x{target_res}"
            }
        }

    async def _download_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        semaphore: asyncio.Semaphore
    ) -> Optional[bytes]:
        async with semaphore:
            for attempt in range(1, 4):
                try:
                    response = await client.get(url, timeout=5.0)
                    response.raise_for_status()
                    return response.content
                except Exception as e:
                    logger.warning(f"Download attempt {attempt} failed for {url}: {str(e)}")
                    if attempt < 3:
                        await asyncio.sleep(attempt * 0.5)
            return None
