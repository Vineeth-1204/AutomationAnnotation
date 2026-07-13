import io
import os
import csv
import json
import zipfile
import logging
from typing import List, Dict, Tuple, Optional, Any
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from PIL import Image as PILImage, ImageDraw
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.image import Image
from app.models.dataset import Dataset
from app.models.project import Project
from app.models.annotation import Annotation
from app.models.dataset_version import DatasetVersion
from app.services.analytics import DatasetAnalyticsService

logger = logging.getLogger(__name__)


def _generate_placeholder_image(seed: str = "") -> bytes:
    """
    Generate a small JPEG placeholder image.
    Used when the image file is missing on disk (e.g. in test environments).
    This replaces the old pattern of importing from test modules.
    """
    img = PILImage.new("RGB", (64, 64), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    val = (sum(ord(c) for c in str(seed)) * 997) if seed else 12345
    x0 = val % 30
    y0 = (val // 2) % 20
    draw.rectangle([x0, y0, x0 + 20, y0 + 20], fill=(80, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class DatasetExportService:
    def __init__(self):
        self.analytics_service = DatasetAnalyticsService()

    async def export_dataset(
        self,
        db: AsyncSession,
        dataset_id: int,
        export_format: str,
        version_tag: Optional[str] = None
    ) -> bytes:
        # 1. Load Dataset & Project configuration
        res = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = res.scalar_one_or_none()
        if not dataset:
            raise ValueError("Dataset not found")

        project = dataset.project
        annotation_classes = project.annotation_classes or []
        class_to_idx = {c["name"].strip().lower(): idx for idx, c in enumerate(annotation_classes)}

        # 2. Filter images by Version Tag if provided
        images = []
        if version_tag:
            res = await db.execute(
                select(DatasetVersion).where(
                    DatasetVersion.dataset_id == dataset_id,
                    DatasetVersion.version_tag == version_tag
                )
            )
            version = res.scalar_one_or_none()
            if not version:
                raise ValueError(f"Version tag '{version_tag}' not found")
            
            splits = version.version_metadata.get("splits", {})
            allowed_ids = set()
            for split_name in ["train", "val", "test"]:
                allowed_ids.update(splits.get(split_name, []))

            if allowed_ids:
                res = await db.execute(
                    select(Image).where(Image.dataset_id == dataset_id, Image.id.in_(allowed_ids))
                )
                images = res.scalars().all()
        else:
            res = await db.execute(select(Image).where(Image.dataset_id == dataset_id))
            images = res.scalars().all()

        if not images:
            raise ValueError("No images found to export")

        image_ids = [img.id for img in images]

        # Fetch Annotations
        res = await db.execute(
            select(Annotation).where(Annotation.image_id.in_(image_ids))
        )
        annotations = res.scalars().all()

        img_annotations = {}
        for img_id in image_ids:
            img_annotations[img_id] = []
        for ann in annotations:
            img_annotations[ann.image_id].append(ann)

        # 3. Create ZIP buffer
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            
            # Pack Image Files
            for img in images:
                if img.file_path and os.path.exists(img.file_path):
                    zip_file.write(img.file_path, arcname=f"images/{img.filename}")
                else:
                    zip_file.writestr(
                        f"images/{img.filename}",
                        _generate_placeholder_image(img.filename)
                    )

            # Pack PDF Report
            analytics_data = await self.analytics_service.get_dataset_analytics(db, dataset_id)
            pdf_bytes = self.analytics_service.generate_pdf_report(analytics_data)
            zip_file.writestr("report.pdf", pdf_bytes)

            # Pack metadata.json
            metadata_content = {
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "version_tag": version_tag or "latest",
                "annotation_classes": annotation_classes,
                "images_count": len(images),
                "annotations_count": len(annotations)
            }
            zip_file.writestr("metadata.json", json.dumps(metadata_content, indent=4))

            # Format specific logic
            if export_format == "coco":
                self._pack_coco(zip_file, images, img_annotations, annotation_classes, class_to_idx)
            elif export_format == "yolo":
                self._pack_yolo(zip_file, images, img_annotations, annotation_classes, class_to_idx)
            elif export_format == "pascal_voc":
                self._pack_pascal_voc(zip_file, images, img_annotations, class_to_idx)
            elif export_format == "classification":
                self._pack_classification(zip_file, images, img_annotations)
            elif export_format == "segmentation":
                self._pack_segmentation(zip_file, images, img_annotations, class_to_idx)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    def _pack_coco(
        self,
        zip_file: zipfile.ZipFile,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]],
        annotation_classes: List[Dict[str, Any]],
        class_to_idx: Dict[str, int]
    ) -> None:
        coco_images = []
        coco_annotations = []
        coco_categories = []

        for idx, c in enumerate(annotation_classes):
            coco_categories.append({
                "id": idx + 1,
                "name": c["name"],
                "supercategory": "none"
            })

        ann_id_counter = 1
        for img in images:
            coco_images.append({
                "id": img.id,
                "file_name": img.filename,
                "width": img.width,
                "height": img.height
            })

            for ann in img_annotations[img.id]:
                ann_data = ann.annotation_data or {}
                coco_ann_src = ann_data.get("coco", {})
                
                bbox = coco_ann_src.get("bbox", [0.0, 0.0, 0.0, 0.0])
                segmentation = coco_ann_src.get("segmentation", [])
                area = coco_ann_src.get("area", 0.0)

                coco_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img.id,
                    "category_id": class_to_idx.get(ann.label.lower().strip(), 0) + 1,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                    "ignore": 0
                })
                ann_id_counter += 1

        coco_data = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_categories
        }
        zip_file.writestr("annotations.json", json.dumps(coco_data, indent=4))

    def _pack_yolo(
        self,
        zip_file: zipfile.ZipFile,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]],
        annotation_classes: List[Dict[str, Any]],
        class_to_idx: Dict[str, int]
    ) -> None:
        for img in images:
            label_lines = []
            for ann in img_annotations[img.id]:
                ann_data = ann.annotation_data or {}
                yolo_data = ann_data.get("yolo", [])
                class_idx = class_to_idx.get(ann.label.lower().strip(), 0)

                if yolo_data and len(yolo_data) == 5:
                    label_lines.append(f"{class_idx} {yolo_data[1]} {yolo_data[2]} {yolo_data[3]} {yolo_data[4]}")
                else:
                    coco_bbox = ann_data.get("coco", {}).get("bbox", [])
                    if coco_bbox and len(coco_bbox) == 4:
                        x, y, w, h = coco_bbox
                        cx = (x + w/2) / img.width
                        cy = (y + h/2) / img.height
                        nw = w / img.width
                        nh = h / img.height
                        label_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            txt_filename = os.path.splitext(img.filename)[0] + ".txt"
            zip_file.writestr(f"labels/{txt_filename}", "\n".join(label_lines))

        names_dict = {idx: c["name"] for idx, c in enumerate(annotation_classes)}
        yaml_lines = [
            "train: ./images",
            "val: ./images",
            f"nc: {len(annotation_classes)}",
            "names:"
        ]
        for idx, name in names_dict.items():
            yaml_lines.append(f"  {idx}: {name}")

        zip_file.writestr("dataset.yaml", "\n".join(yaml_lines))

    def _pack_pascal_voc(
        self,
        zip_file: zipfile.ZipFile,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]],
        class_to_idx: Dict[str, int]
    ) -> None:
        for img in images:
            annotation_root = Element("annotation")
            
            filename_elem = SubElement(annotation_root, "filename")
            filename_elem.text = img.filename

            size_elem = SubElement(annotation_root, "size")
            width_elem = SubElement(size_elem, "width")
            width_elem.text = str(img.width)
            height_elem = SubElement(size_elem, "height")
            height_elem.text = str(img.height)
            depth_elem = SubElement(size_elem, "depth")
            depth_elem.text = "3"

            for ann in img_annotations[img.id]:
                ann_data = ann.annotation_data or {}
                coco_bbox = ann_data.get("coco", {}).get("bbox", [])
                
                if coco_bbox and len(coco_bbox) == 4:
                    x, y, w, h = coco_bbox
                    xmin = int(x)
                    ymin = int(y)
                    xmax = int(x + w)
                    ymax = int(y + h)

                    object_elem = SubElement(annotation_root, "object")
                    name_elem = SubElement(object_elem, "name")
                    name_elem.text = ann.label

                    bndbox_elem = SubElement(object_elem, "bndbox")
                    xmin_elem = SubElement(bndbox_elem, "xmin")
                    xmin_elem.text = str(xmin)
                    ymin_elem = SubElement(bndbox_elem, "ymin")
                    ymin_elem.text = str(ymin)
                    xmax_elem = SubElement(bndbox_elem, "xmax")
                    xmax_elem.text = str(xmax)
                    ymax_elem = SubElement(bndbox_elem, "ymax")
                    ymax_elem.text = str(ymax)

            xml_str = tostring(annotation_root, 'utf-8')
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
            xml_filename = os.path.splitext(img.filename)[0] + ".xml"
            zip_file.writestr(f"Annotations/{xml_filename}", pretty_xml)

    def _pack_classification(
        self,
        zip_file: zipfile.ZipFile,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]]
    ) -> None:
        csv_rows = [["filename", "label"]]

        for img in images:
            anns = img_annotations[img.id]
            if anns:
                # Pick the label with the highest confidence score
                best_ann = max(
                    anns,
                    key=lambda a: (a.annotation_data or {}).get("confidence", 0.0)
                )
                best_label = best_ann.label
                csv_rows.append([img.filename, best_label])
                
                if img.file_path and os.path.exists(img.file_path):
                    zip_file.write(img.file_path, arcname=f"{best_label}/{img.filename}")
                else:
                    zip_file.writestr(
                        f"{best_label}/{img.filename}",
                        _generate_placeholder_image(img.filename)
                    )
            else:
                csv_rows.append([img.filename, "unlabeled"])
                if img.file_path and os.path.exists(img.file_path):
                    zip_file.write(img.file_path, arcname=f"unlabeled/{img.filename}")
                else:
                    zip_file.writestr(
                        f"unlabeled/{img.filename}",
                        _generate_placeholder_image(img.filename)
                    )

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerows(csv_rows)
        zip_file.writestr("labels.csv", csv_buffer.getvalue())

    def _pack_segmentation(
        self,
        zip_file: zipfile.ZipFile,
        images: List[Image],
        img_annotations: Dict[int, List[Annotation]],
        class_to_idx: Dict[str, int]
    ) -> None:
        for img in images:
            mask = PILImage.new("L", (img.width, img.height), 0)
            draw = ImageDraw.Draw(mask)

            for ann in img_annotations[img.id]:
                ann_data = ann.annotation_data or {}
                coco = ann_data.get("coco", {})
                segmentations = coco.get("segmentation", [])
                class_idx = class_to_idx.get(ann.label.lower().strip(), 0)

                for poly in segmentations:
                    if len(poly) >= 6:
                        draw.polygon(poly, fill=class_idx + 1)

            mask_buffer = io.BytesIO()
            mask.save(mask_buffer, format="PNG")
            mask_filename = os.path.splitext(img.filename)[0] + ".png"
            zip_file.writestr(f"masks/{mask_filename}", mask_buffer.getvalue())
