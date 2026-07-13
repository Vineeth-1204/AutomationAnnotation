import os
import shutil
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import asyncio

from app.models.project import Project
from app.models.dataset import Dataset
from app.models.user import User
from app.models.image import Image
from app.models.annotation import Annotation
from app.models.processing_job import ProcessingJob
from tests.api.test_pipeline_and_annotations import make_crisp_image_bytes

@pytest.fixture(scope="function", autouse=True)
def manage_storage_directory():
    storage_dir = os.path.join(os.getcwd(), "storage", "images")
    os.makedirs(storage_dir, exist_ok=True)
    yield
    if os.path.exists("storage"):
        shutil.rmtree("storage")

@pytest.mark.asyncio
async def test_auto_annotation_engine_and_validation(client: AsyncClient, db_session: AsyncSession) -> None:
    # 1. Register and login User
    user_data = {"email": "annotator@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_data)
    res = await client.post("/api/v1/auth/login", data={"username": user_data["email"], "password": user_data["password"]})
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    result = await db_session.execute(select(User).where(User.email == user_data["email"]))
    user = result.scalar_one()

    # 2. Seed Project with annotation classes and aliases
    project_classes = [
        {"name": "car", "aliases": ["automobile", "vehicle"], "color": "#ff0000"},
        {"name": "pedestrian", "aliases": ["person"], "color": "#00ff00"}
    ]
    project = Project(
        name="Auto Annotation Test Proj",
        owner_id=user.id,
        annotation_classes=project_classes
    )
    db_session.add(project)
    await db_session.flush()

    dataset = Dataset(name="Auto Annotation Test Dataset", project_id=project.id, owner_id=user.id)
    db_session.add(dataset)
    await db_session.flush()

    # Write real crisp dummy images to the filesystem
    storage_dir = os.path.join(os.getcwd(), "storage", "images")
    os.makedirs(storage_dir, exist_ok=True)

    # Image A: test NMS (has overlapping boxes in mock output)
    img_a_filename = "test_nms_image.jpg"
    img_a_path = os.path.join(storage_dir, img_a_filename)
    with open(img_a_path, "wb") as f:
        f.write(make_crisp_image_bytes())
    
    img_a = Image(
        filename=img_a_filename,
        file_path=img_a_path,
        width=512,
        height=512,
        dataset_id=dataset.id
    )
    db_session.add(img_a)

    # Image B: test low-confidence rejection
    img_b_filename = "test_conf_image.jpg"
    img_b_path = os.path.join(storage_dir, img_b_filename)
    with open(img_b_path, "wb") as f:
        f.write(make_crisp_image_bytes())
    
    img_b = Image(
        filename=img_b_filename,
        file_path=img_b_path,
        width=512,
        height=512,
        dataset_id=dataset.id
    )
    db_session.add(img_b)

    # Image C: test tiny box rejection
    img_c_filename = "test_tiny_image.jpg"
    img_c_path = os.path.join(storage_dir, img_c_filename)
    with open(img_c_path, "wb") as f:
        f.write(make_crisp_image_bytes())
    
    img_c = Image(
        filename=img_c_filename,
        file_path=img_c_path,
        width=512,
        height=512,
        dataset_id=dataset.id
    )
    db_session.add(img_c)

    # Image D: test normal flow + alias mapping + invalid label filtering
    img_d_filename = "normal_image.jpg"
    img_d_path = os.path.join(storage_dir, img_d_filename)
    with open(img_d_path, "wb") as f:
        f.write(make_crisp_image_bytes())
    
    img_d = Image(
        filename=img_d_filename,
        file_path=img_d_path,
        width=512,
        height=512,
        dataset_id=dataset.id
    )
    db_session.add(img_d)

    await db_session.commit()

    # 3. Trigger Auto Annotation job
    res = await client.post(
        f"/api/v1/datasets/{dataset.id}/annotate",
        headers=headers
    )
    assert res.status_code == 202
    job_id = res.json()["id"]

    # 4. Poll background job completion
    job_status = None
    for _ in range(15):
        await asyncio.sleep(0.5)
        res = await client.get(f"/api/v1/datasets/jobs/{job_id}", headers=headers)
        assert res.status_code == 200
        job_status = res.json()
        if job_status["status"] in ["COMPLETED", "FAILED"]:
            break

    assert job_status["status"] == "COMPLETED"
    assert job_status["result"]["images_processed"] == 4
    assert job_status["result"]["images_failed"] == 0

    # 5. Fetch generated annotations via API
    res = await client.get(
        f"/api/v1/datasets/{dataset.id}/annotations",
        headers=headers
    )
    assert res.status_code == 200
    annotations = res.json()

    # Calculate annotations per image
    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Verification 1: NMS Suppression (Image A)
    # The mock AI returned two overlapping 'car' boxes (one conf 0.95, one conf 0.80).
    # NMS should suppress the 0.80 box, keeping only the 0.95 box.
    img_a_annotations = ann_by_image.get(img_a.id, [])
    assert len(img_a_annotations) == 1
    assert img_a_annotations[0]["label"] == "car"
    assert img_a_annotations[0]["annotation_data"]["confidence"] == 0.95
    # Check COCO and YOLO coordinates formats exist
    assert "coco" in img_a_annotations[0]["annotation_data"]
    assert "yolo" in img_a_annotations[0]["annotation_data"]

    # Verification 2: Confidence Filter (Image B)
    # Mock AI returned a prediction with confidence 0.35, below the 0.50 threshold.
    # It should have been completely discarded.
    img_b_annotations = ann_by_image.get(img_b.id, [])
    assert len(img_b_annotations) == 0

    # Verification 3: Tiny Box Filter (Image C)
    # Mock AI returned a 5x5 bounding box, below the 16x16 minimum size.
    # It should have been completely discarded.
    img_c_annotations = ann_by_image.get(img_c.id, [])
    assert len(img_c_annotations) == 0

    # Verification 4: Alias mapping and Invalid class rejection (Image D)
    # Mock AI returned:
    #  1. 'car' main match (should be accepted -> label 'car')
    #  2. 'automobile' alias match (should be mapped -> label 'car')
    #  3. 'unrelated_noise_class' invalid label (should be discarded)
    # Total annotations should be 4 (2 for 'car' and 2 for 'pedestrian').
    img_d_annotations = ann_by_image.get(img_d.id, [])
    assert len(img_d_annotations) == 4
    labels = [ann["label"] for ann in img_d_annotations]
    assert labels.count("car") == 2
    assert labels.count("pedestrian") == 2
