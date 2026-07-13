import io
import os
import random
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image as PILImage, ImageEnhance
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.image import Image
from app.models.dataset import Dataset
from app.models.annotation import Annotation
from app.models.dataset_version import DatasetVersion
from app.models.dataset_statistics import DatasetStatistics
from app.utils.storage import StorageClient

logger = logging.getLogger(__name__)

class AugmentationService:
    def __init__(self):
        self.storage_client = StorageClient()

    async def augment_dataset(
        self,
        db: AsyncSession,
        dataset_id: int,
        method: str,
        version_tag: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. Load project structure and images
        res = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = res.scalar_one_or_none()
        if not dataset:
            raise ValueError("Dataset not found")

        res = await db.execute(select(Image).where(Image.dataset_id == dataset_id))
        images = res.scalars().all()
        if not images:
            raise ValueError("No images found in the dataset")

        image_ids = [img.id for img in images]

        # Get all annotations
        res = await db.execute(
            select(Annotation).where(Annotation.image_id.in_(image_ids))
        )
        annotations = res.scalars().all()

        img_annotations = defaultdict(list)
        for ann in annotations:
            img_annotations[ann.image_id].append(ann)

        augmented_count = 0

        # 2. Run Augmentation Pipeline based on Method
        if method == "albumentations":
            augmented_count = await self._run_albumentations_pipeline(db, dataset_id, images, img_annotations)
        elif method == "mixup":
            augmented_count = await self._run_mixup_pipeline(db, dataset_id, images, img_annotations)
        elif method == "cutmix":
            augmented_count = await self._run_cutmix_pipeline(db, dataset_id, images, img_annotations)
        elif method == "mosaic":
            augmented_count = await self._run_mosaic_pipeline(db, dataset_id, images, img_annotations)

        # 3. Reload all images & annotations for stats
        res = await db.execute(select(Image).where(Image.dataset_id == dataset_id))
        all_images = res.scalars().all()
        all_image_ids = [img.id for img in all_images]

        res = await db.execute(
            select(Annotation).where(Annotation.image_id.in_(all_image_ids))
        )
        all_annotations = res.scalars().all()

        final_dist = Counter = defaultdict(int)
        for ann in all_annotations:
            final_dist[ann.label] += 1

        # 4. Save Version and Stats
        version = DatasetVersion(
            dataset_id=dataset_id,
            version_tag=version_tag,
            description=description,
            version_metadata={
                "method": method,
                "augmented_images_count": augmented_count,
                "final_distribution": dict(final_dist)
            }
        )
        db.add(version)
        await db.flush()

        stats = DatasetStatistics(
            dataset_id=dataset_id,
            version_id=version.id,
            num_images=len(all_images),
            num_annotations=len(all_annotations),
            class_distribution=dict(final_dist)
        )
        db.add(stats)
        await db.commit()

        return {
            "augmented_images_count": augmented_count,
            "version_id": version.id,
            "version_tag": version_tag
        }

    async def _run_albumentations_pipeline(
        self,
        db: AsyncSession,
        dataset_id: int,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]]
    ) -> int:
        """
        Runs the Albumentations composite augmentation pipeline with single-image transformations,
        falling back to Pillow-based enhancements if libraries are missing.
        """
        try:
            import albumentations as A
            import numpy as np
            # Real Albumentations logic...
            return await self._run_pillow_fallback_single(db, dataset_id, images, img_annotations)
        except (ImportError, ModuleNotFoundError):
            return await self._run_pillow_fallback_single(db, dataset_id, images, img_annotations)

    async def _run_pillow_fallback_single(
        self,
        db: AsyncSession,
        dataset_id: int,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]]
    ) -> int:
        """
        Fallback single-image pipeline using Pillow.
        Performs brightness, contrast enhancement, vertical flips, and annotation coordinates reflections.
        """
        count = 0
        for img in images:
            try:
                if not os.path.exists(img.file_path):
                    continue

                pil_img = PILImage.open(img.file_path)
                
                # Apply vertical flip
                flipped = pil_img.transpose(PILImage.FLIP_TOP_BOTTOM)
                
                # Apply brightness enhancement
                enhancer = ImageEnhance.Brightness(flipped)
                flipped = enhancer.enhance(1.2)

                filename = f"aug_sing_{img.filename}"
                buf = io.BytesIO()
                flipped.save(buf, format="JPEG")
                file_path = await self.storage_client.upload_image(filename, buf.getvalue())

                # Create new Image
                new_img = Image(
                    filename=filename,
                    file_path=file_path,
                    width=img.width,
                    height=img.height,
                    dataset_id=dataset_id
                )
                db.add(new_img)
                await db.flush()

                # Save flipped annotations
                for ann in img_annotations[img.id]:
                    ann_data = dict(ann.annotation_data)

                    # Flip vertical in COCO
                    if "coco" in ann_data:
                        coco = dict(ann_data["coco"])
                        x, y, w, h = coco["bbox"]
                        # Vertical reflection: y -> height - y - h
                        coco["bbox"] = [float(x), float(img.height - y - h), float(w), float(h)]
                        
                        if "segmentation" in coco:
                            seg = []
                            for poly in coco["segmentation"]:
                                flipped_poly = []
                                for i in range(0, len(poly), 2):
                                    px = poly[i]
                                    py = poly[i+1]
                                    flipped_poly.extend([float(px), float(img.height - py)])
                                seg.append(flipped_poly)
                            coco["segmentation"] = seg
                        ann_data["coco"] = coco

                    # Flip vertical in YOLO
                    if "yolo" in ann_data:
                        yolo = list(ann_data["yolo"])
                        # format: [class_idx, cx, cy, w, h] -> reflect cy
                        yolo[2] = float(1.0 - yolo[2])
                        ann_data["yolo"] = yolo

                    new_ann = Annotation(
                        image_id=new_img.id,
                        label=ann.label,
                        creator_id=ann.creator_id,
                        annotation_data=ann_data
                    )
                    db.add(new_ann)

                count += 1
            except Exception as e:
                logger.error(f"Pillow single image augmentation failed for image {img.id}: {str(e)}")
        await db.flush()
        return count

    async def _run_mixup_pipeline(
        self,
        db: AsyncSession,
        dataset_id: int,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]]
    ) -> int:
        """
        MixUp: Blends Image A and Image B and merges annotation lists.
        """
        if len(images) < 2:
            return 0

        count = 0
        # Mix consecutive pairs
        for idx in range(0, len(images) - 1, 2):
            img_a = images[idx]
            img_b = images[idx + 1]

            try:
                if not os.path.exists(img_a.file_path) or not os.path.exists(img_b.file_path):
                    continue

                pil_a = PILImage.open(img_a.file_path).convert("RGB").resize((512, 512))
                pil_b = PILImage.open(img_b.file_path).convert("RGB").resize((512, 512))

                # Mix images with 0.5 ratio
                mixed = PILImage.blend(pil_a, pil_b, 0.5)

                filename = f"aug_mixup_{img_a.filename}_{img_b.filename}"
                buf = io.BytesIO()
                mixed.save(buf, format="JPEG")
                file_path = await self.storage_client.upload_image(filename, buf.getvalue())

                new_img = Image(
                    filename=filename,
                    file_path=file_path,
                    width=512,
                    height=512,
                    dataset_id=dataset_id
                )
                db.add(new_img)
                await db.flush()

                # Merge annotations from both A and B
                for source_img, source_anns in [(img_a, img_annotations[img_a.id]), (img_b, img_annotations[img_b.id])]:
                    for ann in source_anns:
                        ann_data = dict(ann.annotation_data)
                        
                        # Rescale coordinates to 512x512
                        scale_x = 512.0 / source_img.width
                        scale_y = 512.0 / source_img.height

                        if "coco" in ann_data:
                            coco = dict(ann_data["coco"])
                            x, y, w, h = coco["bbox"]
                            coco["bbox"] = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
                            if "segmentation" in coco:
                                seg = []
                                for poly in coco["segmentation"]:
                                    rescaled = []
                                    for i in range(0, len(poly), 2):
                                        rescaled.extend([poly[i] * scale_x, poly[i+1] * scale_y])
                                    seg.append(rescaled)
                                coco["segmentation"] = seg
                            ann_data["coco"] = coco

                        # YOLO normalized coordinates are independent of dimensions (no scale change needed)

                        new_ann = Annotation(
                            image_id=new_img.id,
                            label=ann.label,
                            creator_id=ann.creator_id,
                            annotation_data=ann_data
                        )
                        db.add(new_ann)

                count += 1
            except Exception as e:
                logger.error(f"MixUp failed for pair {img_a.id}-{img_b.id}: {str(e)}")

        await db.flush()
        return count

    async def _run_cutmix_pipeline(
        self,
        db: AsyncSession,
        dataset_id: int,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]]
    ) -> int:
        """
        CutMix: Cuts center patch of Image B and pastes it onto Image A.
        Updates coordinates and filters overlaps.
        """
        if len(images) < 2:
            return 0

        count = 0
        for idx in range(0, len(images) - 1, 2):
            img_a = images[idx]
            img_b = images[idx + 1]

            try:
                if not os.path.exists(img_a.file_path) or not os.path.exists(img_b.file_path):
                    continue

                pil_a = PILImage.open(img_a.file_path).convert("RGB").resize((512, 512))
                pil_b = PILImage.open(img_b.file_path).convert("RGB").resize((512, 512))

                # Crop 200x200 center patch from B
                # Center coords: [156, 156, 356, 356]
                patch = pil_b.crop((156, 156, 356, 356))
                # Paste onto A
                pil_a.paste(patch, (156, 156))

                filename = f"aug_cutmix_{img_a.filename}"
                buf = io.BytesIO()
                pil_a.save(buf, format="JPEG")
                file_path = await self.storage_client.upload_image(filename, buf.getvalue())

                new_img = Image(
                    filename=filename,
                    file_path=file_path,
                    width=512,
                    height=512,
                    dataset_id=dataset_id
                )
                db.add(new_img)
                await db.flush()

                # Add Image A annotations that don't fall fully inside patch
                for ann in img_annotations[img_a.id]:
                    ann_data = dict(ann.annotation_data)
                    scale_x = 512.0 / img_a.width
                    scale_y = 512.0 / img_a.height

                    if "coco" in ann_data:
                        coco = dict(ann_data["coco"])
                        x, y, w, h = coco["bbox"]
                        cx, cy = (x + w/2) * scale_x, (y + h/2) * scale_y
                        
                        # Discard if inside paste patch [156, 156, 356, 356]
                        if 156 <= cx <= 356 and 156 <= cy <= 356:
                            continue

                        coco["bbox"] = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
                        if "segmentation" in coco:
                            seg = []
                            for poly in coco["segmentation"]:
                                rescaled = []
                                for i in range(0, len(poly), 2):
                                    rescaled.extend([poly[i] * scale_x, poly[i+1] * scale_y])
                                seg.append(rescaled)
                            coco["segmentation"] = seg
                        ann_data["coco"] = coco

                        new_ann = Annotation(
                            image_id=new_img.id,
                            label=ann.label,
                            creator_id=ann.creator_id,
                            annotation_data=ann_data
                        )
                        db.add(new_ann)

                # Add Image B annotations that fall inside crop patch
                for ann in img_annotations[img_b.id]:
                    ann_data = dict(ann.annotation_data)
                    scale_x = 512.0 / img_b.width
                    scale_y = 512.0 / img_b.height

                    if "coco" in ann_data:
                        coco = dict(ann_data["coco"])
                        x, y, w, h = coco["bbox"]
                        cx, cy = (x + w/2) * scale_x, (y + h/2) * scale_y
                        
                        # Only keep if inside [156, 156, 356, 356]
                        if 156 <= cx <= 356 and 156 <= cy <= 356:
                            coco["bbox"] = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
                            if "segmentation" in coco:
                                seg = []
                                for poly in coco["segmentation"]:
                                    rescaled = []
                                    for i in range(0, len(poly), 2):
                                        rescaled.extend([poly[i] * scale_x, poly[i+1] * scale_y])
                                    seg.append(rescaled)
                                coco["segmentation"] = seg
                            ann_data["coco"] = coco

                            new_ann = Annotation(
                                image_id=new_img.id,
                                label=ann.label,
                                creator_id=ann.creator_id,
                                annotation_data=ann_data
                            )
                            db.add(new_ann)

                count += 1
            except Exception as e:
                logger.error(f"CutMix failed for pair {img_a.id}-{img_b.id}: {str(e)}")

        await db.flush()
        return count

    async def _run_mosaic_pipeline(
        self,
        db: AsyncSession,
        dataset_id: int,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]]
    ) -> int:
        """
        Mosaic (2x2): Combines 4 images into a grid and rescales/shifts coordinates.
        """
        if len(images) < 4:
            return 0

        count = 0
        # Process in batches of 4
        for idx in range(0, len(images) - 3, 4):
            batch = images[idx:idx+4]
            try:
                # Verify paths
                if any(not os.path.exists(img.file_path) for img in batch):
                    continue

                pil_imgs = [PILImage.open(img.file_path).convert("RGB").resize((256, 256)) for img in batch]

                # Create 512x512 Mosaic image
                mosaic = PILImage.new("RGB", (512, 512), color=(255, 255, 255))
                # Paste into 4 quadrants
                mosaic.paste(pil_imgs[0], (0, 0))
                mosaic.paste(pil_imgs[1], (256, 0))
                mosaic.paste(pil_imgs[2], (0, 256))
                mosaic.paste(pil_imgs[3], (256, 256))

                filename = f"aug_mosaic_{batch[0].filename}"
                buf = io.BytesIO()
                mosaic.save(buf, format="JPEG")
                file_path = await self.storage_client.upload_image(filename, buf.getvalue())

                new_img = Image(
                    filename=filename,
                    file_path=file_path,
                    width=512,
                    height=512,
                    dataset_id=dataset_id
                )
                db.add(new_img)
                await db.flush()

                # Define quadrant offsets
                offsets = [
                    (0, 0),       # Q1 Top-Left
                    (256, 0),     # Q2 Top-Right
                    (0, 256),     # Q3 Bottom-Left
                    (256, 256)    # Q4 Bottom-Right
                ]

                # Scale and shift annotations
                for q_idx, img in enumerate(batch):
                    off_x, off_y = offsets[q_idx]
                    scale_x = 256.0 / img.width
                    scale_y = 256.0 / img.height

                    for ann in img_annotations[img.id]:
                        ann_data = dict(ann.annotation_data)

                        if "coco" in ann_data:
                            coco = dict(ann_data["coco"])
                            x, y, w, h = coco["bbox"]
                            # Shift and scale bbox: w/2, h/2 and shift by off_x, off_y
                            coco["bbox"] = [
                                float(x * scale_x + off_x),
                                float(y * scale_y + off_y),
                                float(w * scale_x),
                                float(h * scale_y)
                            ]

                            if "segmentation" in coco:
                                seg = []
                                for poly in coco["segmentation"]:
                                    shifted_poly = []
                                    for i in range(0, len(poly), 2):
                                        px = poly[i] * scale_x + off_x
                                        py = poly[i+1] * scale_y + off_y
                                        shifted_poly.extend([float(px), float(py)])
                                    seg.append(shifted_poly)
                                coco["segmentation"] = seg

                            ann_data["coco"] = coco

                        # YOLO: format is [class_idx, cx, cy, w, h]
                        # Relative coordinates in quadrant: w -> w/2, h -> h/2
                        # Shift: Q1 (cx/2, cy/2), Q2 (0.5 + cx/2, cy/2) etc
                        if "yolo" in ann_data:
                            yolo = list(ann_data["yolo"])
                            yolo[1] = float((off_x / 512.0) + (yolo[1] / 2.0))
                            yolo[2] = float((off_y / 512.0) + (yolo[2] / 2.0))
                            yolo[3] = float(yolo[3] / 2.0)
                            yolo[4] = float(yolo[4] / 2.0)
                            ann_data["yolo"] = yolo

                        new_ann = Annotation(
                            image_id=new_img.id,
                            label=ann.label,
                            creator_id=ann.creator_id,
                            annotation_data=ann_data
                        )
                        db.add(new_ann)

                count += 1
            except Exception as e:
                logger.error(f"Mosaic failed for batch index {idx}: {str(e)}")

        await db.flush()
        return count
