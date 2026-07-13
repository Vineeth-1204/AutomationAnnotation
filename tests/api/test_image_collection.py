import os
import shutil
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.project import Project
from app.models.dataset import Dataset
from app.models.user import User

@pytest.mark.asyncio
async def test_image_collection_flow(client: AsyncClient, db_session: AsyncSession) -> None:
    # 1. Register and login User
    user_data = {"email": "collector@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_data)
    res = await client.post("/api/v1/auth/login", data={"username": user_data["email"], "password": user_data["password"]})
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Get User ID
    result = await db_session.execute(select(User).where(User.email == user_data["email"]))
    user = result.scalar_one()

    # 3. Seed Project and Dataset directly in the DB
    project = Project(name="Test Auto Collection", owner_id=user.id)
    db_session.add(project)
    await db_session.flush()

    dataset = Dataset(name="Test Birds", project_id=project.id, owner_id=user.id)
    db_session.add(dataset)
    await db_session.flush()
    await db_session.commit()

    # 4. Trigger image collection
    payload = {
        "queries": ["pikachu"],
        "limit_per_query": 2
    }
    
    from unittest.mock import patch, MagicMock
    import httpx

    original_get = httpx.AsyncClient.get

    async def mock_get(self, url, *args, **kwargs):
        if str(url).startswith("http://test") or str(url).startswith("/"):
            return await original_get(self, url, *args, **kwargs)
        
        from tests.api.test_pipeline_and_annotations import make_crisp_image_bytes
        content = make_crisp_image_bytes(url)
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = content
        mock_response.raise_for_status = MagicMock()
        return mock_response

    patcher = patch("httpx.AsyncClient.get", mock_get)
    patcher.start()
    
    try:
        res = await client.post(
            f"/api/v1/datasets/{dataset.id}/collect",
            json=payload,
            headers=headers
        )
        assert res.status_code == 202
        job_info = res.json()
        assert job_info["job_type"] == "image_collection"
        assert job_info["dataset_id"] == dataset.id
        job_id = job_info["id"]

        # 5. Poll / inspect background task execution
        import asyncio
        job_status = None
        for _ in range(10):
            await asyncio.sleep(0.5)
            res = await client.get(f"/api/v1/datasets/jobs/{job_id}", headers=headers)
            assert res.status_code == 200
            job_status = res.json()
            if job_status["status"] in ["COMPLETED", "FAILED"]:
                break
                
        assert job_status is not None
        assert job_status["status"] == "COMPLETED"
        assert job_status["result"]["downloaded_count"] == 2
        assert job_status["result"]["failed_count"] == 0
    finally:
        patcher.stop()

    # 6. Verify image files exist locally
    storage_dir = os.path.join(os.getcwd(), "storage", "images")
    assert os.path.exists(storage_dir)
    files = os.listdir(storage_dir)
    assert len(files) >= 2

    # Clean up storage folder after tests
    if os.path.exists("storage"):
        shutil.rmtree("storage")
