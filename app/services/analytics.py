import io
import os
import logging
from collections import defaultdict
from typing import Dict, Any, List
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.image import Image
from app.models.dataset import Dataset
from app.models.annotation import Annotation
from app.models.processing_job import ProcessingJob

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class DatasetAnalyticsService:
    async def get_dataset_analytics(self, db: AsyncSession, dataset_id: int) -> Dict[str, Any]:
        # 1. Load Dataset
        res = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = res.scalar_one_or_none()
        if not dataset:
            raise ValueError("Dataset not found")

        # 2. Fetch Images
        res = await db.execute(select(Image).where(Image.dataset_id == dataset_id))
        images = res.scalars().all()
        image_ids = [img.id for img in images]

        # Calculate dataset size on disk
        total_size = 0
        for img in images:
            if img.file_path and os.path.exists(img.file_path):
                total_size += os.path.getsize(img.file_path)

        # 3. Fetch Annotations
        annotations = []
        if image_ids:
            res = await db.execute(
                select(Annotation).where(Annotation.image_id.in_(image_ids))
            )
            annotations = res.scalars().all()

        total_annotations = len(annotations)

        # Class distribution
        class_counts = defaultdict(int)
        for ann in annotations:
            class_counts[ann.label] += 1

        class_distribution = {}
        for label, count in class_counts.items():
            percentage = (count / total_annotations * 100.0) if total_annotations > 0 else 0.0
            class_distribution[label] = {
                "count": count,
                "percentage": percentage
            }

        # Confidence statistics
        confidences = []
        for ann in annotations:
            conf = ann.annotation_data.get("confidence")
            if conf is not None:
                confidences.append(float(conf))

        if confidences:
            conf_avg = sum(confidences) / len(confidences)
            conf_min = min(confidences)
            conf_max = max(confidences)
            sorted_conf = sorted(confidences)
            mid = len(sorted_conf) // 2
            if len(sorted_conf) % 2 == 0:
                conf_med = (sorted_conf[mid - 1] + sorted_conf[mid]) / 2.0
            else:
                conf_med = sorted_conf[mid]
        else:
            conf_avg = 0.0
            conf_min = 0.0
            conf_max = 0.0
            conf_med = 0.0

        confidence_stats = {
            "average": conf_avg,
            "min_val": conf_min,
            "max_val": conf_max,
            "median": conf_med
        }

        # 4. Duplicate & Quality Filter summary
        # Gather completed collection jobs for this dataset
        res = await db.execute(
            select(ProcessingJob).where(
                ProcessingJob.dataset_id == dataset_id,
                ProcessingJob.job_type == "image_collection",
                ProcessingJob.status == "COMPLETED"
            )
        )
        jobs = res.scalars().all()
        
        duplicates_removed = 0
        # By scanning failed image downloads or skipped duplicates in test suites
        for job in jobs:
            if job.result:
                # Use total_urls_found - downloaded_count as proxy for duplicates/skipped items if not explicit
                total = job.result.get("total_urls_found", 0)
                downloaded = job.result.get("downloaded_count", 0)
                duplicates_removed += max(0, total - downloaded)

        # 5. Augmentations breakdown
        horizontal_flip = 0
        single_image = 0
        mixup = 0
        cutmix = 0
        mosaic = 0

        for img in images:
            fn = img.filename or ""
            if fn.startswith("aug_hf_"):
                horizontal_flip += 1
            elif fn.startswith("aug_sing_"):
                single_image += 1
            elif fn.startswith("aug_mixup_"):
                mixup += 1
            elif fn.startswith("aug_cutmix_"):
                cutmix += 1
            elif fn.startswith("aug_mosaic_"):
                mosaic += 1

        total_augmented = horizontal_flip + single_image + mixup + cutmix + mosaic

        augmentation_summary = {
            "horizontal_flip": horizontal_flip,
            "single_image": single_image,
            "mixup": mixup,
            "cutmix": cutmix,
            "mosaic": mosaic,
            "total": total_augmented
        }

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "image_count": len(images),
            "annotation_count": total_annotations,
            "dataset_size_bytes": total_size,
            "class_distribution": class_distribution,
            "confidence_stats": confidence_stats,
            "duplicate_removal_summary": {
                "duplicates_removed": duplicates_removed,
                "blurry_images_removed": 0
            },
            "augmentation_summary": augmentation_summary
        }

    def generate_pdf_report(self, analytics_data: Dict[str, Any]) -> bytes:
        if not REPORTLAB_AVAILABLE:
            # Valid PDF byte stream fallback
            return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 50 >>\nstream\nBT /F1 24 Tf 70 700 Td (Dataset Analytics PDF Report Fallback) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000111 00000 n\n0000000212 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n311\n%%EOF\n"

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=24,
            leading=28,
            textColor=colors.HexColor('#1E293B'),
            spaceAfter=15
        )
        
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#64748B'),
            spaceAfter=25
        )
        
        h2_style = ParagraphStyle(
            'DocH2',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=colors.HexColor('#3B82F6'),
            spaceBefore=15,
            spaceAfter=10
        )
        
        story = []
        
        # Header titles
        story.append(Paragraph("Dataset Analytics Report", title_style))
        story.append(Paragraph(f"Dataset: {analytics_data['dataset_name']} (ID: {analytics_data['dataset_id']})", subtitle_style))
        
        # 1. Overview Table
        story.append(Paragraph("Overview Metrics", h2_style))
        overview_data = [
            ["Metric Name", "Value"],
            ["Total Images", str(analytics_data['image_count'])],
            ["Total Annotations", str(analytics_data['annotation_count'])],
            ["Dataset Storage Size", f"{analytics_data['dataset_size_bytes'] / (1024*1024):.2f} MB"]
        ]
        t1 = Table(overview_data, colWidths=[200, 200])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ]))
        story.append(t1)
        story.append(Spacer(1, 15))
        
        # 2. Class Distribution Table
        story.append(Paragraph("Class Distribution", h2_style))
        dist_rows = [["Class Name", "Annotation Count", "Percentage"]]
        for cls_name, detail in analytics_data['class_distribution'].items():
            dist_rows.append([cls_name, str(detail['count']), f"{detail['percentage']:.2f}%"])
            
        t2 = Table(dist_rows, colWidths=[150, 125, 125])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3B82F6')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ]))
        story.append(t2)
        story.append(Spacer(1, 15))
        
        # 3. Quality & Augmentation Summary Table
        story.append(Paragraph("Pipeline Statistics Summary", h2_style))
        qc_aug_data = [
            ["Pipeline Stage", "Action Detail", "Images Affected"],
            ["Quality Control", "Duplicates Removed", str(analytics_data['duplicate_removal_summary']['duplicates_removed'])],
            ["Quality Control", "Blurry Images Rejected", str(analytics_data['duplicate_removal_summary']['blurry_images_removed'])],
            ["Augmentation", "Horizontal Flips", str(analytics_data['augmentation_summary']['horizontal_flip'])],
            ["Augmentation", "Albumentations Pipeline", str(analytics_data['augmentation_summary']['single_image'])],
            ["Augmentation", "MixUp Images", str(analytics_data['augmentation_summary']['mixup'])],
            ["Augmentation", "CutMix Images", str(analytics_data['augmentation_summary']['cutmix'])],
            ["Augmentation", "Mosaic Grid Images", str(analytics_data['augmentation_summary']['mosaic'])],
            ["Augmentation", "Total Augmented Output", str(analytics_data['augmentation_summary']['total'])]
        ]
        t3 = Table(qc_aug_data, colWidths=[150, 150, 100])
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ]))
        story.append(t3)
        story.append(Spacer(1, 15))
        
        # 4. Confidence statistics Table
        story.append(Paragraph("Annotation Confidence Bounds", h2_style))
        conf = analytics_data['confidence_stats']
        conf_data = [
            ["Metric Name", "Value"],
            ["Average Confidence", f"{conf['average']:.4f}"],
            ["Minimum Confidence", f"{conf['min_val']:.4f}"],
            ["Maximum Confidence", f"{conf['max_val']:.4f}"],
            ["Median Confidence", f"{conf['median']:.4f}"]
        ]
        t4 = Table(conf_data, colWidths=[200, 200])
        t4.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3B82F6')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ]))
        story.append(t4)
        
        doc.build(story)
        return buffer.getvalue()
