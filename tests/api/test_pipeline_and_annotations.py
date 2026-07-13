import os
import shutil
import pytest
import httpx
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from unittest.mock import patch, MagicMock, AsyncMock

from app.models.project import Project
from app.models.dataset import Dataset
from app.models.user import User
from app.models.processing_job import ProcessingJob
from app.models.image import Image

def make_solid_image_bytes():
    from PIL import Image as PILImage
    import io
    img = PILImage.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def make_crisp_image_bytes(url: str = ""):
    from PIL import Image as PILImage, ImageDraw
    import io
    img = PILImage.new("RGB", (100, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    url_val = (sum(ord(c) for c in str(url)) * 997) if url else 12345
    x0 = url_val % 40
    y0 = (url_val // 2) % 30
    x1 = x0 + 20 + (url_val % 30)
    y1 = y0 + 20 + (url_val % 30)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

@pytest.mark.asyncio
async def test_image_preprocessing_pipeline_filters(client: AsyncClient, db_session: AsyncSession) -> None:
    # Set up user and project
    user_data = {"email": "pipeline@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_data)
    res = await client.post("/api/v1/auth/login", data={"username": user_data["email"], "password": user_data["password"]})
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    result = await db_session.execute(select(User).where(User.email == user_data["email"]))
    user = result.scalar_one()

    project = Project(name="Pipeline Test Proj", owner_id=user.id)
    db_session.add(project)
    await db_session.flush()

    dataset = Dataset(name="Pipeline Test Dataset", project_id=project.id, owner_id=user.id)
    db_session.add(dataset)
    await db_session.flush()
    await db_session.commit()

    # 1. Test Blur Rejection (solid image has 0 variance)
    original_get = httpx.AsyncClient.get
    async def mock_get_blurry(self, url, *args, **kwargs):
        if str(url).startswith("http://test") or str(url).startswith("/"):
            return await original_get(self, url, *args, **kwargs)
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.content = make_solid_image_bytes()
        mock_res.raise_for_status = MagicMock()
        return mock_res

    with patch("httpx.AsyncClient.get", mock_get_blurry):
        res = await client.post(
            f"/api/v1/datasets/{dataset.id}/collect",
            json={"queries": ["blurry"], "limit_per_query": 1},
            headers=headers
        )
        assert res.status_code == 202
        job_id = res.json()["id"]

        import asyncio
        for _ in range(10):
            await asyncio.sleep(0.3)
            status_res = await client.get(f"/api/v1/datasets/jobs/{job_id}", headers=headers)
            if status_res.json()["status"] == "COMPLETED":
                break
        
        # Blurry image should have been rejected (downloaded_count = 0, failed_count = 1)
        job_data = status_res.json()
        assert job_data["status"] == "COMPLETED"
        assert job_data["result"]["downloaded_count"] == 0
        assert job_data["result"]["failed_count"] == 1

    # 2. Test Crisp Image Acceptance
    async def mock_get_crisp(self, url, *args, **kwargs):
        if str(url).startswith("http://test") or str(url).startswith("/"):
            return await original_get(self, url, *args, **kwargs)
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.content = make_crisp_image_bytes(url)
        mock_res.raise_for_status = MagicMock()
        return mock_res

    # Use patch context to run crisp test
    with patch("httpx.AsyncClient.get", mock_get_crisp):
        res = await client.post(
            f"/api/v1/datasets/{dataset.id}/collect",
            json={"queries": ["crisp"], "limit_per_query": 1},
            headers=headers
        )
        assert res.status_code == 202
        job_id = res.json()["id"]

        for _ in range(10):
            await asyncio.sleep(0.3)
            status_res = await client.get(f"/api/v1/datasets/jobs/{job_id}", headers=headers)
            if status_res.json()["status"] == "COMPLETED":
                break
        
        # Crisp image should have been accepted and normalized
        job_data = status_res.json()
        assert job_data["status"] == "COMPLETED"
        assert job_data["result"]["downloaded_count"] == 1
        assert job_data["result"]["failed_count"] == 0

    # Clean up local storage
    if os.path.exists("storage"):
        shutil.rmtree("storage")


@pytest.mark.asyncio
async def test_annotation_classes_workflow_and_validation(client: AsyncClient, db_session: AsyncSession) -> None:
    # Set up user and project
    user_data = {"email": "annotation@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_data)
    res = await client.post("/api/v1/auth/login", data={"username": user_data["email"], "password": user_data["password"]})
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    result = await db_session.execute(select(User).where(User.email == user_data["email"]))
    user = result.scalar_one()

    project = Project(name="Annotation Project", owner_id=user.id)
    db_session.add(project)
    await db_session.flush()

    dataset = Dataset(name="Annotation Dataset", project_id=project.id, owner_id=user.id)
    db_session.add(dataset)
    await db_session.flush()
    await db_session.commit()

    # 1. Try to submit annotation classes before collection jobs are created (should fail with 400 Bad Request)
    payload = {
        "classes": [
            {"name": "car", "aliases": ["automobile"], "color": "#ff0000"}
        ]
    }
    res = await client.post(
        f"/api/v1/projects/{project.id}/annotation-classes",
        json=payload,
        headers=headers
    )
    assert res.status_code == 400
    assert "No collection jobs found" in res.json()["error"]["message"]

    # 2. Add an incomplete processing job (should fail with 400 Bad Request)
    job = ProcessingJob(
        job_type="image_collection",
        status="RUNNING",
        dataset_id=dataset.id,
        creator_id=user.id
    )
    db_session.add(job)
    await db_session.commit()

    res = await client.post(
        f"/api/v1/projects/{project.id}/annotation-classes",
        json=payload,
        headers=headers
    )
    assert res.status_code == 400
    assert "Image collection is not complete" in res.json()["error"]["message"]

    # 3. Complete the job
    job.status = "COMPLETED"
    db_session.add(job)
    await db_session.commit()

    # 4. Try submitting invalid name configurations (should fail with 422 validation error)
    invalid_payload = {
        "classes": [
            {"name": "car-invalid", "aliases": [], "color": "#ff0000"}
        ]
    }
    res = await client.post(
        f"/api/v1/projects/{project.id}/annotation-classes",
        json=invalid_payload,
        headers=headers
    )
    assert res.status_code == 422

    # 5. Try submitting duplicate classes configuration (should fail with 422 validation error)
    duplicate_payload = {
        "classes": [
            {"name": "car", "aliases": [], "color": "#ff0000"},
            {"name": "car", "aliases": [], "color": "#00ff00"}
        ]
    }
    res = await client.post(
        f"/api/v1/projects/{project.id}/annotation-classes",
        json=duplicate_payload,
        headers=headers
    )
    assert res.status_code == 422

    # 6. Submit valid configuration (should succeed)
    valid_payload = {
        "classes": [
            {"name": "car", "aliases": ["automobile", "vehicle"], "color": "#ff0000"},
            {"name": "pedestrian", "aliases": ["person"], "color": "#00ff00"}
        ]
    }
    res = await client.post(
        f"/api/v1/projects/{project.id}/annotation-classes",
        json=valid_payload,
        headers=headers
    )
    assert res.status_code == 200
    project_info = res.json()
    assert len(project_info["annotation_classes"]) == 2
    assert project_info["annotation_classes"][0]["name"] == "car"
    assert "automobile" in project_info["annotation_classes"][0]["aliases"]

    # 7. Test retrieving configured classes via GET
    res = await client.get(
        f"/api/v1/projects/{project.id}/annotation-classes",
        headers=headers
    )
    assert res.status_code == 200
    classes = res.json()
    assert len(classes) == 2
    assert classes[0]["name"] == "car"
    assert classes[1]["name"] == "pedestrian"
