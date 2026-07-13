import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_register_and_login(client: AsyncClient) -> None:
    # 1. Register a new user
    register_data = {
        "email": "testuser@example.com",
        "password": "strongpassword123",
        "full_name": "Test User",
    }
    response = await client.post("/api/v1/users/", json=register_data)
    assert response.status_code == 201
    user_json = response.json()
    assert user_json["email"] == register_data["email"]
    assert user_json["full_name"] == register_data["full_name"]
    assert "id" in user_json
    
    # 2. Login
    login_data = {
        "username": "testuser@example.com",
        "password": "strongpassword123",
    }
    response = await client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code == 200
    token_json = response.json()
    assert "access_token" in token_json
    assert "refresh_token" in token_json
    assert token_json["token_type"] == "bearer"
    
    # 3. Test token endpoint
    headers = {"Authorization": f"Bearer {token_json['access_token']}"}
    response = await client.post("/api/v1/auth/test-token", headers=headers)
    assert response.status_code == 200
    me_json = response.json()
    assert me_json["email"] == register_data["email"]

@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient) -> None:
    login_data = {
        "username": "nonexistent@example.com",
        "password": "wrongpassword",
    }
    response = await client.post("/api/v1/auth/login", data=login_data)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AuthenticationError"

@pytest.mark.asyncio
async def test_refresh_token(client: AsyncClient) -> None:
    # 1. Register & Login
    email = "refresh@example.com"
    pwd = "password123"
    await client.post("/api/v1/users/", json={"email": email, "password": pwd})
    
    res = await client.post("/api/v1/auth/login", data={"username": email, "password": pwd})
    assert res.status_code == 200
    tokens = res.json()
    refresh_tok = tokens["refresh_token"]
    
    # 2. Refresh access token
    res = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_tok})
    assert res.status_code == 200
    new_tokens = res.json()
    assert "access_token" in new_tokens
    assert new_tokens["refresh_token"] == refresh_tok

@pytest.mark.asyncio
async def test_email_verification_and_reset(client: AsyncClient) -> None:
    # 1. Register a user (is_verified = False)
    email = "verify@example.com"
    pwd = "password123"
    res = await client.post("/api/v1/users/", json={"email": email, "password": pwd})
    assert res.status_code == 201
    assert res.json()["is_verified"] is False
    
    # 2. Generate verification token
    from app.core.security import create_verification_token
    token = create_verification_token(email)
    
    # 3. Confirm email verification
    res = await client.post("/api/v1/auth/verify-email", json={"token": token})
    assert res.status_code == 200
    assert res.json()["is_verified"] is True

@pytest.mark.asyncio
async def test_role_based_access_control(client: AsyncClient) -> None:
    # 1. Create a regular user
    user_data = {"email": "regular@example.com", "password": "password123", "role": "user"}
    res = await client.post("/api/v1/users/", json=user_data)
    assert res.status_code == 201
    user_id = res.json()["id"]
    
    # Login as regular user
    res = await client.post("/api/v1/auth/login", data={"username": "regular@example.com", "password": "password123"})
    token_regular = res.json()["access_token"]
    
    # 2. Try to delete the user using their own regular token (Should be 403 Forbidden)
    res = await client.delete(f"/api/v1/users/{user_id}", headers={"Authorization": f"Bearer {token_regular}"})
    assert res.status_code == 403
    
    # 3. Create an admin user
    admin_data = {"email": "admin@example.com", "password": "password123", "role": "admin"}
    res = await client.post("/api/v1/users/", json=admin_data)
    assert res.status_code == 201
    
    res = await client.post("/api/v1/auth/login", data={"username": "admin@example.com", "password": "password123"})
    token_admin = res.json()["access_token"]
    
    # Delete the regular user using admin token (Should be 200 OK)
    res = await client.delete(f"/api/v1/users/{user_id}", headers={"Authorization": f"Bearer {token_admin}"})
    assert res.status_code == 200
