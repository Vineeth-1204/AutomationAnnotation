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
async def test_dataset_balancing_and_stratification(client: AsyncClient, db_session: AsyncSession) -> None:
    # 1. Register and login User
    user_data = {"email": "balancer@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_data)
    res = await client.post("/api/v1/auth/login", data={"username": user_data["email"], "password": user_data["password"]})
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    result = await db_session.execute(select(User).where(User.email == user_data["email"]))
    user = result.scalar_one()

    # 2. Seed Project and Dataset
    project_classes = [
        {"name": "car", "aliases": ["automobile"], "color": "#ff0000"},
        {"name": "pedestrian", "aliases": [], "color": "#00ff00"}
    ]
    project = Project(name="Balancing Test Proj", owner_id=user.id, annotation_classes=project_classes)
    db_session.add(project)
    await db_session.flush()

    dataset = Dataset(name="Balancing Test Dataset", project_id=project.id, owner_id=user.id)
    db_session.add(dataset)
    await db_session.flush()

    # Write dummy images on disk
    storage_dir = os.path.join(os.getcwd(), "storage", "images")
    os.makedirs(storage_dir, exist_ok=True)

    # We create 3 images:
    # Image 1: contains 1 'car' + 1 'pedestrian'
    img1_path = os.path.join(storage_dir, "img1.jpg")
    with open(img1_path, "wb") as f:
        f.write(make_crisp_image_bytes())
    img1 = Image(filename="img1.jpg", file_path=img1_path, width=512, height=512, dataset_id=dataset.id)
    db_session.add(img1)

    # Image 2: contains 1 'car'
    img2_path = os.path.join(storage_dir, "img2.jpg")
    with open(img2_path, "wb") as f:
        f.write(make_crisp_image_bytes())
    img2 = Image(filename="img2.jpg", file_path=img2_path, width=512, height=512, dataset_id=dataset.id)
    db_session.add(img2)

    # Image 3: contains 1 'car'
    img3_path = os.path.join(storage_dir, "img3.jpg")
    with open(img3_path, "wb") as f:
        f.write(make_crisp_image_bytes())
    img3 = Image(filename="img3.jpg", file_path=img3_path, width=512, height=512, dataset_id=dataset.id)
    db_session.add(img3)

    await db_session.flush()

    # Add Annotations
    # Image 1 annotations
    ann1 = Annotation(
        image_id=img1.id, label="car", creator_id=user.id,
        annotation_data={"confidence": 0.90, "coco": {"bbox": [10.0, 10.0, 100.0, 100.0], "segmentation": [[10.0, 10.0]]}, "yolo": [0, 0.1, 0.1, 0.2, 0.2]}
    )
    ann2 = Annotation(
        image_id=img1.id, label="pedestrian", creator_id=user.id,
        annotation_data={"confidence": 0.85, "coco": {"bbox": [150.0, 150.0, 50.0, 50.0], "segmentation": [[150.0, 150.0, 200.0, 150.0, 200.0, 200.0, 150.0, 200.0]]}, "yolo": [1, 0.34, 0.34, 0.1, 0.1]}
    )
    db_session.add_all([ann1, ann2])

    # Image 2 annotation
    ann3 = Annotation(
        image_id=img2.id, label="car", creator_id=user.id,
        annotation_data={"confidence": 0.88, "coco": {"bbox": [20.0, 20.0, 80.0, 80.0]}, "yolo": [0, 0.12, 0.12, 0.16, 0.16]}
    )
    db_session.add(ann3)

    # Image 3 annotation
    ann4 = Annotation(
        image_id=img3.id, label="car", creator_id=user.id,
        annotation_data={"confidence": 0.92, "coco": {"bbox": [30.0, 30.0, 90.0, 90.0]}, "yolo": [0, 0.15, 0.15, 0.18, 0.18]}
    )
    db_session.add(ann4)

    await db_session.commit()

    # 3. Call balance-and-split route
    payload = {
        "oversample": True,
        "imbalance_ratio": 0.50,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "version_tag": "v1.0.0-balanced",
        "description": "Balanced dataset using horizontal flip augmentations"
    }

    res = await client.post(
        f"/api/v1/datasets/{dataset.id}/balance-and-split",
        json=payload,
        headers=headers
    )
    assert res.status_code == 200
    data = res.json()

    # 4. Verify response statistics
    assert data["initial_distribution"] == {"car": 3, "pedestrian": 1}
    assert data["imbalance_identified"] is True
    assert data["minority_classes"] == ["pedestrian"]
    # Flipped Image 1 containing minority class 'pedestrian'
    assert data["augmented_images_count"] == 1
    assert data["final_distribution"] == {"car": 4, "pedestrian": 2}
    
    # Check splits list
    splits = data["splits"]
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    total_split_len = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_split_len == 4  # 3 original + 1 augmented image

    # 5. Verify database records
    # Check if augmented image exists in DB
    result = await db_session.execute(
        select(Image).where(Image.dataset_id == dataset.id, Image.filename.startswith("aug_hf_"))
    )
    aug_img = result.scalar_one_or_none()
    assert aug_img is not None
    assert aug_img.width == 512
    assert aug_img.height == 512

    # Verify annotation coordinate flips on the new image
    result = await db_session.execute(
        select(Annotation).where(Annotation.image_id == aug_img.id)
    )
    aug_anns = result.scalars().all()
    assert len(aug_anns) == 2

    # Map by class label
    aug_ann_by_label = {ann.label: ann for ann in aug_anns}
    
    # Original pedestrian bbox was [150.0, 150.0, 50.0, 50.0]
    # Flipped pedestrian bbox should be [512 - 150 - 50, 150, 50, 50] = [312.0, 150.0, 50.0, 50.0]
    aug_ped = aug_ann_by_label["pedestrian"]
    assert aug_ped.annotation_data["coco"]["bbox"] == [312.0, 150.0, 50.0, 50.0]
    
    # Original pedestrian polygon: [150, 150, 200, 150, 200, 200, 150, 200]
    # Flipped pedestrian polygon: [512 - x, y] -> [362, 150, 312, 150, 312, 200, 362, 200]
    assert aug_ped.annotation_data["coco"]["segmentation"][0] == [362.0, 150.0, 312.0, 150.0, 312.0, 200.0, 362.0, 200.0]

    # Original YOLO center_x was 0.34
    # Flipped YOLO center_x should be 1.0 - 0.34 = 0.66
    assert aug_ped.annotation_data["yolo"][1] == pytest.approx(0.66)

    # Verify Version and Statistics are saved
    result = await db_session.execute(
        select(DatasetVersion).where(DatasetVersion.dataset_id == dataset.id, DatasetVersion.version_tag == "v1.0.0-balanced")
    )
    version = result.scalar_one_or_none()
    assert version is not None
    assert version.description == "Balanced dataset using horizontal flip augmentations"
    assert "splits" in version.version_metadata

    result = await db_session.execute(
        select(DatasetStatistics).where(DatasetStatistics.version_id == version.id)
    )
    stats = result.scalar_one_or_none()
    assert stats is not None
    assert stats.num_images == 4
    assert stats.num_annotations == 6
    assert stats.class_distribution == {"car": 4, "pedestrian": 2}
