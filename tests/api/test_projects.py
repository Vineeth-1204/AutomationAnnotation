import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_project_crud_and_ownership(client: AsyncClient) -> None:
    # 1. Register and login User A
    user_a = {"email": "usera_projects@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_a)
    res = await client.post("/api/v1/auth/login", data={"username": user_a["email"], "password": user_a["password"]})
    token_a = res.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # 2. Register and login User B
    user_b = {"email": "userb_projects@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_b)
    res = await client.post("/api/v1/auth/login", data={"username": user_b["email"], "password": user_b["password"]})
    token_b = res.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 3. User A creates a Project
    project_payload = {
        "name": "Self-Driving Car Dataset",
        "dataset_description": "Collect objects on urban roads",
        "user_prompt": "Gather highway traffic footage",
        "annotation_classes": ["car", "pedestrian", "traffic_light"],
        "desired_image_count": 250,
        "dataset_type": "object_detection",
        "output_format": "yolo",
        "status": "pending",
    }
    res = await client.post("/api/v1/projects/", json=project_payload, headers=headers_a)
    assert res.status_code == 201
    project_a = res.json()
    assert project_a["name"] == project_payload["name"]
    assert project_a["desired_image_count"] == 250
    assert project_a["annotation_classes"] == project_payload["annotation_classes"]
    project_id = project_a["id"]

    # 4. User B attempts to read User A's project (403 Forbidden)
    res = await client.get(f"/api/v1/projects/{project_id}", headers=headers_b)
    assert res.status_code == 403

    # 5. User A reads their own project (200 OK)
    res = await client.get(f"/api/v1/projects/{project_id}", headers=headers_a)
    assert res.status_code == 200
    assert res.json()["name"] == project_payload["name"]

    # 6. User A updates their project status
    update_payload = {"status": "completed", "desired_image_count": 300}
    res = await client.put(f"/api/v1/projects/{project_id}", json=update_payload, headers=headers_a)
    assert res.status_code == 200
    updated_project = res.json()
    assert updated_project["status"] == "completed"
    assert updated_project["desired_image_count"] == 300

    # 7. User A lists their projects
    res = await client.get("/api/v1/projects/", headers=headers_a)
    assert res.status_code == 200
    assert len(res.json()) == 1
    assert res.json()[0]["id"] == project_id

    # 8. User B lists projects (should see 0 projects)
    res = await client.get("/api/v1/projects/", headers=headers_b)
    assert res.status_code == 200
    assert len(res.json()) == 0

    # 9. User A deletes their project
    res = await client.delete(f"/api/v1/projects/{project_id}", headers=headers_a)
    assert res.status_code == 200
    
    # 10. Verify project is deleted
    res = await client.get(f"/api/v1/projects/{project_id}", headers=headers_a)
    assert res.status_code == 404
