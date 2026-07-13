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
from tests.api.test_pipeline_and_annotations import make_crisp_image_bytes

@pytest.fixture(scope="function", autouse=True)
def manage_storage_directory():
    storage_dir = os.path.join(os.getcwd(), "storage", "images")
    os.makedirs(storage_dir, exist_ok=True)
    yield
    if os.path.exists("storage"):
        shutil.rmtree("storage")

@pytest.mark.asyncio
async def test_dataset_analytics_and_pdf_reports(client: AsyncClient, db_session: AsyncSession) -> None:
    # 1. Register and login User
    user_data = {"email": "analyst@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_data)
    res = await client.post("/api/v1/auth/login", data={"username": user_data["email"], "password": user_data["password"]})
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    result = await db_session.execute(select(User).where(User.email == user_data["email"]))
    user = result.scalar_one()

    # 2. Seed Project and Dataset
    project_classes = [
        {"name": "car", "aliases": [], "color": "#ff0000"},
        {"name": "pedestrian", "aliases": [], "color": "#00ff00"}
    ]
    project = Project(name="Analytics Test Proj", owner_id=user.id, annotation_classes=project_classes)
    db_session.add(project)
    await db_session.flush()

    dataset = Dataset(name="Analytics Test Dataset", project_id=project.id, owner_id=user.id)
    db_session.add(dataset)
    await db_session.flush()

    # Write dummy images on disk
    storage_dir = os.path.join(os.getcwd(), "storage", "images")
    os.makedirs(storage_dir, exist_ok=True)

    # Image 1 (augmented filename starting with aug_hf_)
    img1_filename = "aug_hf_img1.jpg"
    img1_path = os.path.join(storage_dir, img1_filename)
    with open(img1_path, "wb") as f:
        f.write(make_crisp_image_bytes(img1_filename))
    img1 = Image(filename=img1_filename, file_path=img1_path, width=512, height=512, dataset_id=dataset.id)
    db_session.add(img1)

    # Image 2 (augmented filename starting with aug_sing_)
    img2_filename = "aug_sing_img2.jpg"
    img2_path = os.path.join(storage_dir, img2_filename)
    with open(img2_path, "wb") as f:
        f.write(make_crisp_image_bytes(img2_filename))
    img2 = Image(filename=img2_filename, file_path=img2_path, width=512, height=512, dataset_id=dataset.id)
    db_session.add(img2)

    await db_session.flush()

    # 3. Add Annotations with explicit confidence values to verify stats
    ann1 = Annotation(
        image_id=img1.id, label="car", creator_id=user.id,
        annotation_data={"confidence": 0.80, "coco": {"bbox": [10, 10, 50, 50]}, "yolo": [0, 0.1, 0.1, 0.1, 0.1]}
    )
    ann2 = Annotation(
        image_id=img2.id, label="pedestrian", creator_id=user.id,
        annotation_data={"confidence": 0.90, "coco": {"bbox": [20, 20, 40, 40]}, "yolo": [1, 0.2, 0.2, 0.1, 0.1]}
    )
    db_session.add_all([ann1, ann2])
    await db_session.commit()

    # 4. Trigger JSON Analytics Endpoint
    res = await client.get(
        f"/api/v1/datasets/{dataset.id}/analytics",
        headers=headers
    )
    assert res.status_code == 200
    data = res.json()

    # Check metrics
    assert data["dataset_id"] == dataset.id
    assert data["dataset_name"] == "Analytics Test Dataset"
    assert data["image_count"] == 2
    assert data["annotation_count"] == 2
    assert data["dataset_size_bytes"] > 0

    # Check class distribution
    assert "car" in data["class_distribution"]
    assert "pedestrian" in data["class_distribution"]
    assert data["class_distribution"]["car"]["count"] == 1
    assert data["class_distribution"]["car"]["percentage"] == 50.0

    # Check confidence stats
    assert data["confidence_stats"]["average"] == pytest.approx(0.85)
    assert data["confidence_stats"]["min_val"] == pytest.approx(0.80)
    assert data["confidence_stats"]["max_val"] == pytest.approx(0.90)
    assert data["confidence_stats"]["median"] == pytest.approx(0.85)

    # Check augmentations breakdown
    assert data["augmentation_summary"]["horizontal_flip"] == 1
    assert data["augmentation_summary"]["single_image"] == 1
    assert data["augmentation_summary"]["total"] == 2

    # 5. Trigger PDF Report Endpoint
    res = await client.get(
        f"/api/v1/datasets/{dataset.id}/analytics/pdf",
        headers=headers
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF-")
