import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_user_profile_and_items(client: AsyncClient) -> None:
    # 1. Register and login User A
    user_a_register = {
        "email": "usera@example.com",
        "password": "usera12345",
        "full_name": "User A",
    }
    res = await client.post("/api/v1/users/", json=user_a_register)
    assert res.status_code == 201
    
    res = await client.post("/api/v1/auth/login", data={"username": "usera@example.com", "password": "usera12345"})
    assert res.status_code == 200
    token_a = res.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}
    
    # 2. Register and login User B
    user_b_register = {
        "email": "userb@example.com",
        "password": "userb12345",
        "full_name": "User B",
    }
    res = await client.post("/api/v1/users/", json=user_b_register)
    assert res.status_code == 201
    
    res = await client.post("/api/v1/auth/login", data={"username": "userb@example.com", "password": "userb12345"})
    assert res.status_code == 200
    token_b = res.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}
    
    # 3. Create an item for User A
    item_in = {"title": "User A Item", "description": "Description A"}
    res = await client.post("/api/v1/items/", json=item_in, headers=headers_a)
    assert res.status_code == 201
    item_a_json = res.json()
    assert item_a_json["title"] == "User A Item"
    item_a_id = item_a_json["id"]
    
    # 4. User B attempts to read User A's item (Should be 403 Forbidden)
    res = await client.get(f"/api/v1/items/{item_a_id}", headers=headers_b)
    assert res.status_code == 403
    
    # 5. User A reads their own item (Should be 200)
    res = await client.get(f"/api/v1/items/{item_a_id}", headers=headers_a)
    assert res.status_code == 200
    assert res.json()["title"] == "User A Item"
    
    # 6. Read User A items list
    res = await client.get("/api/v1/items/", headers=headers_a)
    assert res.status_code == 200
    assert len(res.json()) == 1
    assert res.json()[0]["title"] == "User A Item"
