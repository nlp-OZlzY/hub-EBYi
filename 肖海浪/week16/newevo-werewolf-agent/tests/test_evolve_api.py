"""自演化 API 测试"""

import pytest
from unittest.mock import patch, AsyncMock
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from api.server import app
from api import evolve_service


@pytest.fixture(autouse=True)
def clear_jobs():
    evolve_service.evolve_jobs.clear()
    yield
    evolve_service.evolve_jobs.clear()


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_start_evolve_job(client):
    with patch("api.evolve_service.run_evolve_job", new_callable=AsyncMock):
        resp = await client.post(
            "/evolve/jobs",
            json={"rounds": 2, "config_name": "simple_4", "shuffle": False},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] in ("pending", "running")


@pytest.mark.asyncio
async def test_list_role_prompts(client):
    resp = await client.get("/prompts/roles")
    assert resp.status_code == 200
    data = resp.json()
    assert "werewolf" in data
    assert "content" in data["werewolf"]
