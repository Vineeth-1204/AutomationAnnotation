import io
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.image import Image
from app.models.dataset import Dataset
from app.models.annotation import Annotation
from app.models.dataset_version import DatasetVersion
from app.models.dataset_statistics import DatasetStatistics
from app.repositories.image import ImageRepository
from app.repositories.annotation import AnnotationRepository
from app.utils.storage import StorageClient

logger = logging.getLogger(__name__)

class DatasetBalancerService:
    def __init__(self):
        self.storage_client = StorageClient()

    async def balance_and_split(
        self,
        db: AsyncSession,
        dataset_id: int,
        oversample: bool,
        imbalance_ratio: float,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
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
        image_ids = [img.id for img in images]

        if not images:
            raise ValueError("No images found in the dataset")

        # 2. Get all annotations for the dataset
        res = await db.execute(
            select(Annotation).where(Annotation.image_id.in_(image_ids))
        )
        annotations = res.scalars().all()

        # Build mappings
        img_annotations = defaultdict(list)
        for ann in annotations:
            img_annotations[ann.image_id].append(ann)

        # Count initial class distribution
        initial_counts = Counter()
        for ann in annotations:
            initial_counts[ann.label] += 1

        initial_dist = dict(initial_counts)

        # 3. Identify Minority Classes
        imbalance_identified = False
        minority_classes = []
        augmented_count = 0

        if initial_dist:
            max_class, max_count = initial_counts.most_common(1)[0]
            for label, count in initial_dist.items():
                if count < imbalance_ratio * max_count:
                    minority_classes.append(label)
            if minority_classes:
                imbalance_identified = True

        # 4. Augmentation / Oversampling (Horizontal Flip)
        if oversample and imbalance_identified and minority_classes:
            # We locate original images containing minority annotations and augment them
            target_images = []
            for img in images:
                img_anns = img_annotations[img.id]
                has_minority = any(ann.label in minority_classes for ann in img_anns)
                if has_minority:
                    target_images.append(img)

            for img in target_images:
                try:
                    # Open original image
                    if not os.path.exists(img.file_path):
                        logger.warning(f"Original image file missing for augmentation: {img.file_path}")
                        continue

                    pil_img = PILImage.open(img.file_path)
                    
                    # Flip horizontally
                    flipped_img = pil_img.transpose(PILImage.FLIP_LEFT_RIGHT)
                    
                    # Upload to storage
                    aug_filename = f"aug_hf_{img.filename}"
                    buf = io.BytesIO()
                    flipped_img.save(buf, format="JPEG")
                    aug_file_path = await self.storage_client.upload_image(aug_filename, buf.getvalue())

                    # Save new Image record
                    new_img = Image(
                        filename=aug_filename,
                        file_path=aug_file_path,
                        width=img.width,
                        height=img.height,
                        dataset_id=dataset_id
                    )
                    db.add(new_img)
                    await db.flush()

                    # Duplicate and flip annotation coordinates
                    for ann in img_annotations[img.id]:
                        ann_data = dict(ann.annotation_data)

                        # Flip COCO coordinates
                        if "coco" in ann_data:
                            coco = dict(ann_data["coco"])
                            x, y, w, h = coco["bbox"]
                            coco["bbox"] = [float(img.width - x - w), float(y), float(w), float(h)]
                            
                            if "segmentation" in coco:
                                seg = []
                                for poly in coco["segmentation"]:
                                    flipped_poly = []
                                    for i in range(0, len(poly), 2):
                                        px = poly[i]
                                        py = poly[i+1]
                                        flipped_poly.extend([float(img.width - px), float(py)])
                                    seg.append(flipped_poly)
                                coco["segmentation"] = seg
                            ann_data["coco"] = coco

                        # Flip YOLO coordinates
                        if "yolo" in ann_data:
                            yolo = list(ann_data["yolo"])
                            # [class_idx, cx, cy, w, h] -> reflect cx
                            yolo[1] = float(1.0 - yolo[1])
                            ann_data["yolo"] = yolo

                        new_ann = Annotation(
                            image_id=new_img.id,
                            label=ann.label,
                            creator_id=ann.creator_id,
                            annotation_data=ann_data
                        )
                        db.add(new_ann)

                    augmented_count += 1
                except Exception as e:
                    logger.error(f"Augmentation failed for image {img.id}: {str(e)}")

            # Flush augmented images and annotations to DB
            await db.flush()

        # Re-fetch all images & annotations for stratified splitting
        res = await db.execute(select(Image).where(Image.dataset_id == dataset_id))
        all_images = res.scalars().all()
        all_image_ids = [img.id for img in all_images]

        res = await db.execute(
            select(Annotation).where(Annotation.image_id.in_(all_image_ids))
        )
        all_annotations = res.scalars().all()

        # Map annotations for final counts
        all_img_labels = defaultdict(list)
        final_counts = Counter()
        for ann in all_annotations:
            all_img_labels[ann.image_id].append(ann.label)
            final_counts[ann.label] += 1

        final_dist = dict(final_counts)

        # 5. Greedy Multilabel Stratified Split
        train_ids, val_ids, test_ids = self._stratified_split(
            all_images_labels={img.id: all_img_labels[img.id] for img in all_images},
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

        splits_data = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }

        # 6. Save Version and Stats
        version = DatasetVersion(
            dataset_id=dataset_id,
            version_tag=version_tag,
            description=description,
            version_metadata={
                "splits": splits_data,
                "initial_distribution": initial_dist,
                "final_distribution": final_dist,
                "augmented_images_count": augmented_count
            }
        )
        db.add(version)
        await db.flush()

        stats = DatasetStatistics(
            dataset_id=dataset_id,
            version_id=version.id,
            num_images=len(all_images),
            num_annotations=len(all_annotations),
            class_distribution=final_dist
        )
        db.add(stats)
        await db.commit()

        return {
            "initial_distribution": initial_dist,
            "imbalance_identified": imbalance_identified,
            "minority_classes": minority_classes,
            "augmented_images_count": augmented_count,
            "final_distribution": final_dist,
            "splits": splits_data,
            "version_id": version.id,
            "version_tag": version_tag
        }

    def _stratified_split(
        self,
        all_images_labels: Dict[int, List[str]],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Tuple[List[int], List[int], List[int]]:
        total_ratios = [train_ratio, val_ratio, test_ratio]
        splits = [[], [], []]
        
        # Calculate label counts
        label_counts = Counter()
        for labels in all_images_labels.values():
            label_counts.update(labels)

        # Class counts in each split
        split_class_counts = [Counter() for _ in range(3)]

        # Sort images by rarest class
        def get_rarest_count(item):
            img_id, labels = item
            if not labels:
                return 999999
            return min(label_counts[lbl] for lbl in labels)

        sorted_images = sorted(all_images_labels.items(), key=get_rarest_count)

        for img_id, labels in sorted_images:
            if not labels:
                # Distribute empty images based on split capacities
                t_len = len(splits[0])
                v_len = len(splits[1])
                tot = t_len + v_len + len(splits[2])
                if tot == 0:
                    splits[0].append(img_id)
                else:
                    if (t_len / tot) < train_ratio:
                        splits[0].append(img_id)
                    elif (v_len / tot) < val_ratio:
                        splits[1].append(img_id)
                    else:
                        splits[2].append(img_id)
                continue

            # Greedy allocation based on current split ratios deficits
            deficits = []
            for i in range(3):
                ratio = total_ratios[i]
                split_total = sum(split_class_counts[i].values())
                
                deficit = 0.0
                for lbl in labels:
                    expected = ratio
                    current = (split_class_counts[i][lbl] / split_total) if split_total > 0 else 0.0
                    deficit += (expected - current)
                deficits.append(deficit)

            best_idx = deficits.index(max(deficits))
            splits[best_idx].append(img_id)
            split_class_counts[best_idx].update(labels)

        return splits[0], splits[1], splits[2]
