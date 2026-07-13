import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import app.db.session as db_session
from app.models.image import Image
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.annotation import Annotation
from app.models.processing_job import ProcessingJob
from app.repositories.annotation import AnnotationRepository
from app.repositories.processing_job import ProcessingJobRepository
from app.services.base import BaseService
from app.services.annotation_engine import AutoAnnotationEngine

logger = logging.getLogger(__name__)

class AnnotationService(BaseService[Annotation]):
    def __init__(self, repository: AnnotationRepository):
        super().__init__(repository)
        self.repository = repository
        self.engine = AutoAnnotationEngine()

    async def start_annotation_job(
        self,
        *,
        db: AsyncSession,
        dataset_id: int,
        creator_id: Optional[int] = None
    ) -> ProcessingJob:
        job_repo = ProcessingJobRepository(db)
        job_data = {
            "job_type": "auto_annotation",
            "status": "PENDING",
            "parameters": {
                "dataset_id": dataset_id
            },
            "dataset_id": dataset_id,
            "creator_id": creator_id
        }
        job = await job_repo.create(job_data)
        return job

    async def run_annotation_background_task(
        self,
        job_id: int,
        dataset_id: int
    ) -> None:
        logger.info(f"Starting background auto-annotation job {job_id} for dataset {dataset_id}")
        
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

        # 1. Fetch project annotation classes configuration
        annotation_classes = []
        async with db_session.SessionLocal() as db:
            res = await db.execute(
                select(Project)
                .join(Dataset)
                .where(Dataset.id == dataset_id)
            )
            project = res.scalar_one_or_none()
            if project:
                annotation_classes = project.annotation_classes or []

        # Create mapping of class name to list index for YOLO format class_idx representation
        class_to_idx = {c["name"].strip().lower(): idx for idx, c in enumerate(annotation_classes)}

        # 2. Fetch images in dataset
        images = []
        async with db_session.SessionLocal() as db:
            res = await db.execute(
                select(Image).where(Image.dataset_id == dataset_id)
            )
            images = res.scalars().all()

        annotations_to_create = []
        processed_count = 0
        failed_count = 0
        total_annotations_created = 0

        for image in images:
            try:
                # Open image file
                if not os.path.exists(image.file_path):
                    logger.warning(f"Image file not found: {image.file_path}")
                    failed_count += 1
                    continue

                img = PILImage.open(image.file_path)
                w, h = img.size

                # 3. Model inference: detect candidate predictions
                candidates = await self.engine.run_inference(img, annotation_classes, image.filename)

                # 4. Pipeline validation steps
                # Validate labels and resolve aliases
                valid_preds = self.engine.sanitize_labels_and_aliases(candidates, annotation_classes)
                # Filter by confidence threshold (0.50 default)
                valid_preds = self.engine.filter_by_confidence(valid_preds, threshold=0.50)
                # Filter out tiny boxes (min_dim = 16, min_area = 256)
                valid_preds = self.engine.filter_tiny_boxes(valid_preds, min_dim=16, min_area=256)
                # Apply non-maximum suppression (NMS) to clear overlapping duplicates (IoU 0.50 threshold)
                valid_preds = self.engine.apply_nms(valid_preds, iou_threshold=0.50)

                # 5. Format annotations and collect
                for pred in valid_preds:
                    bbox = pred["bbox"] # [x, y, w, h] in pixels
                    label = pred["label"]
                    class_idx = class_to_idx.get(label.strip().lower(), 0)

                    coco_data = self.engine.generate_coco_data(bbox)
                    yolo_data = self.engine.generate_yolo_data(bbox, w, h, class_idx)

                    annotation_data = {
                        "confidence": float(pred["confidence"]),
                        "coco": coco_data,
                        "yolo": yolo_data
                    }

                    annotations_to_create.append({
                        "image_id": image.id,
                        "label": label,
                        "creator_id": job.creator_id,
                        "annotation_data": annotation_data
                    })
                    total_annotations_created += 1

                processed_count += 1
            except Exception as e:
                logger.error(f"Error annotating image {image.id}: {str(e)}")
                failed_count += 1

        # 6. Save annotations to database
        async with db_session.SessionLocal() as db:
            annotation_repo = AnnotationRepository(db)
            for annotation_in in annotations_to_create:
                await annotation_repo.create(annotation_in)

            job_repo = ProcessingJobRepository(db)
            job = await job_repo.get(job_id)
            
            result_stats = {
                "images_processed": processed_count,
                "images_failed": failed_count,
                "annotations_created": total_annotations_created
            }
            
            await job_repo.update(job, {
                "status": "COMPLETED",
                "completed_at": datetime.utcnow(),
                "result": result_stats
            })
            await db.commit()

        logger.info(f"Background auto-annotation job {job_id} completed successfully")
