import os
import shutil
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.project import Project
from app.models.dataset import Dataset
from app.models.user import User
from app.models.image import Image
from app.models.annotation import Annotation
from app.models.dataset_version import DatasetVersion
from app.models.dataset_statistics import DatasetStatistics
from tests.api.test_pipeline_and_annotations import make_crisp_image_bytes

@pytest.fixture(scope="function", autouse=True)
def manage_storage_directory():
    storage_dir = os.path.join(os.getcwd(), "storage", "images")
    os.makedirs(storage_dir, exist_ok=True)
    yield
    if os.path.exists("storage"):
        shutil.rmtree("storage")

@pytest.mark.asyncio
async def test_dataset_augmentations_flow(client: AsyncClient, db_session: AsyncSession) -> None:
    # 1. Register and login User
    user_data = {"email": "augmenter@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_data)
    res = await client.post("/api/v1/auth/login", data={"username": user_data["email"], "password": user_data["password"]})
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    result = await db_session.execute(select(User).where(User.email == user_data["email"]))
    user = result.scalar_one()

    # 2. Seed Project and Dataset
    project_classes = [
        {"name": "car", "aliases": [], "color": "#ff0000"}
    ]
    project = Project(name="Augment Test Proj", owner_id=user.id, annotation_classes=project_classes)
    db_session.add(project)
    await db_session.flush()

    dataset = Dataset(name="Augment Test Dataset", project_id=project.id, owner_id=user.id)
    db_session.add(dataset)
    await db_session.flush()

    # Write 4 dummy images on disk
    storage_dir = os.path.join(os.getcwd(), "storage", "images")
    os.makedirs(storage_dir, exist_ok=True)

    img_ids = []
    for i in range(1, 5):
        img_name = f"img{i}.jpg"
        img_path = os.path.join(storage_dir, img_name)
        with open(img_path, "wb") as f:
            f.write(make_crisp_image_bytes(img_name))
        
        img = Image(filename=img_name, file_path=img_path, width=512, height=512, dataset_id=dataset.id)
        db_session.add(img)
        await db_session.flush()
        img_ids.append(img.id)

        # Add 1 annotation per image
        # Box coordinates: [10.0 * i, 10.0 * i, 100.0, 100.0]
        ann = Annotation(
            image_id=img.id, label="car", creator_id=user.id,
            annotation_data={
                "confidence": 0.90,
                "coco": {
                    "bbox": [float(10.0 * i), float(10.0 * i), 100.0, 100.0],
                    "segmentation": [[float(10.0 * i), float(10.0 * i)]]
                },
                "yolo": [0, 0.1 * i, 0.1 * i, 0.2, 0.2]
            }
        )
        db_session.add(ann)

    await db_session.commit()

    # Test 1: Single Image Augmentations (Albumentations / Pillow Fallback)
    payload_sing = {
        "method": "albumentations",
        "version_tag": "v1.0.0-sing",
        "description": "Single-image flips and brightness adjustments"
    }
    res = await client.post(f"/api/v1/datasets/{dataset.id}/augment", json=payload_sing, headers=headers)
    assert res.status_code == 200
    assert res.json()["augmented_images_count"] == 4

    # Verify vertical coordinate flip for Image 1:
    # Original bbox: [10.0, 10.0, 100.0, 100.0]
    # Flipped vertical: [10.0, 512 - 10 - 100, 100, 100] = [10.0, 402.0, 100.0, 100.0]
    result = await db_session.execute(
        select(Image).where(Image.dataset_id == dataset.id, Image.filename == "aug_sing_img1.jpg")
    )
    sing_img = result.scalar_one_or_none()
    assert sing_img is not None

    result = await db_session.execute(
        select(Annotation).where(Annotation.image_id == sing_img.id)
    )
    sing_anns = result.scalars().all()
    assert len(sing_anns) == 1
    assert sing_anns[0].annotation_data["coco"]["bbox"] == [10.0, 402.0, 100.0, 100.0]

    # Test 2: Mosaic Augmentation (batches of 4)
    payload_mosaic = {
        "method": "mosaic",
        "version_tag": "v1.0.0-mosaic",
        "description": "2x2 grid mosaic"
    }
    res = await client.post(f"/api/v1/datasets/{dataset.id}/augment", json=payload_mosaic, headers=headers)
    assert res.status_code == 200
    assert res.json()["augmented_images_count"] == 2

    # Verify Mosaic coordinate shifts
    result = await db_session.execute(
        select(Image).where(Image.dataset_id == dataset.id, Image.filename.startswith("aug_mosaic_"))
    )
    mosaic_imgs = result.scalars().all()
    assert len(mosaic_imgs) == 2
    mosaic_img = mosaic_imgs[0]
    assert mosaic_img is not None

    result = await db_session.execute(
        select(Annotation).where(Annotation.image_id == mosaic_img.id)
    )
    mosaic_anns = result.scalars().all()
    assert len(mosaic_anns) == 4

    # Sort annotations by their YOLO centers to match quadrants
    mosaic_anns = sorted(mosaic_anns, key=lambda a: a.annotation_data["yolo"][1])
    
    # Q1 (top-left): Original bbox [10.0, 10.0, 100.0, 100.0] rescaled to 0.5 -> [5.0, 5.0, 50.0, 50.0]
    assert mosaic_anns[0].annotation_data["coco"]["bbox"] == [5.0, 5.0, 50.0, 50.0]

    # Q2 (top-right): Original bbox [20.0, 20.0, 100.0, 100.0] -> scale 0.5 + shift off_x=256 -> [256 + 10, 10, 50, 50] = [266.0, 10.0, 50.0, 50.0]
    assert mosaic_anns[2].annotation_data["coco"]["bbox"] == [266.0, 10.0, 50.0, 50.0]

    # Test 3: MixUp Augmentation (pairs)
    payload_mix = {
        "method": "mixup",
        "version_tag": "v1.0.0-mixup",
        "description": "MixUp pixel blending"
    }
    res = await client.post(f"/api/v1/datasets/{dataset.id}/augment", json=payload_mix, headers=headers)
    assert res.status_code == 200
    assert res.json()["augmented_images_count"] == 5

    # Test 4: CutMix Augmentation (pairs)
    payload_cut = {
        "method": "cutmix",
        "version_tag": "v1.0.0-cutmix",
        "description": "CutMix patch overlays"
    }
    res = await client.post(f"/api/v1/datasets/{dataset.id}/augment", json=payload_cut, headers=headers)
    assert res.status_code == 200
    assert res.json()["augmented_images_count"] == 7
