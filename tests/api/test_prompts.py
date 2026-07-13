import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_understand_prompt_endpoint(client: AsyncClient) -> None:
    # 1. Register and login
    user_data = {"email": "prompt_test@example.com", "password": "password123"}
    await client.post("/api/v1/users/", json=user_data)
    res = await client.post("/api/v1/auth/login", data={"username": user_data["email"], "password": user_data["password"]})
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Test prompt understanding (using the local fallback analyzer)
    prompt_payload = {"prompt": "yellow sports cars on a highway"}
    res = await client.post("/api/v1/prompts/understand", json=prompt_payload, headers=headers)
    assert res.status_code == 200
    
    data = res.json()
    assert data["original_prompt"] == prompt_payload["prompt"]
    assert "optimized_search_queries" in data
    assert "synonyms" in data
    assert "related_keywords" in data
    assert "search_variations" in data
    assert "detected_classes" in data
    
    # Verify fallback has extracted words and generated variations
    assert "car" in data["detected_classes"] or "cars" in data["detected_classes"]
    assert len(data["search_variations"]) > 0
    assert len(data["optimized_search_queries"]) > 0
