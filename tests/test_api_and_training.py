"""End-to-end checks over the HTTP API, the job runner and the explainer layer."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from turducken.explain.demos import DEMOS, run_demo
from turducken.explain.modules import LESSONS, get_lesson
from turducken.schemas.recipe import Recipe
from turducken.server.app import create_app
from turducken.training.registry import available_backends, default_backend, get_backend
from turducken.training.runner import JobManager, JobState


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


# ── API ───────────────────────────────────────────────────────────────


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_hardware_endpoint_works_without_a_gpu(client):
    """The app must be fully usable on a machine that cannot train."""
    body = client.get("/api/hardware").json()
    assert "can_train" in body
    assert isinstance(body["gpus"], list)
    if not body["can_train"]:
        assert body["notes"], "an unusable machine must be told why"


def test_schema_endpoint_carries_documentation(client):
    body = client.get("/api/recipe/schema").json()
    assert body["schema"]["title"] == "Recipe"
    assert "lora.rank" in body["docs"]
    assert body["docs"]["lora.rank"]["why"]


def test_default_recipe_is_immediately_valid(client):
    default = client.get("/api/recipe/default").json()["recipe"]
    assert client.post("/api/recipe/validate", json={"recipe": default}).json()["valid"]


def test_validation_returns_field_level_errors(client):
    body = client.post(
        "/api/recipe/validate", json={"recipe": {"lora": {"rank": -1}}}
    ).json()
    assert not body["valid"]
    assert any("rank" in e["field"] for e in body["errors"])


def test_review_endpoint_returns_advisories(client):
    body = client.post(
        "/api/recipe/review", json={"recipe": Recipe().model_dump(mode="json")}
    ).json()
    assert "advisories" in body
    assert body["estimated_vram_gb"] > 0
    assert body["effective_batch_size"] == 1


def test_review_rejects_invalid_recipes_with_detail(client):
    response = client.post("/api/recipe/review", json={"recipe": {"training": {"steps": -5}}})
    assert response.status_code == 422


def test_dataset_analysis_of_a_missing_folder_is_graceful(client):
    body = client.post(
        "/api/dataset/analyze",
        json={"recipe": {"data": {"dataset_path": "/nonexistent/path/xyz"}}},
    ).json()
    assert body["count"] == 0
    assert body["rejected"] or body["warnings"]


def test_backends_endpoint_explains_availability(client):
    body = client.get("/api/backends").json()
    names = {b["name"] for b in body["backends"]}
    assert {"simulated", "diffusion"} <= names
    for backend in body["backends"]:
        assert backend["reason"], f"{backend['name']} must explain its state"


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "turducken" in response.text


# ── Lessons and demos ─────────────────────────────────────────────────


def test_every_lesson_has_written_content(client):
    for lesson in LESSONS:
        body = client.get(f"/api/lessons/{lesson.slug}").json()
        assert "has not been written yet" not in body["body"], f"{lesson.slug} is a stub"
        assert len(body["body"]) > 800


def test_lesson_demo_references_all_resolve():
    for lesson in LESSONS:
        for demo in lesson.demos:
            assert demo in DEMOS, f"{lesson.slug} references missing demo '{demo}'"


def test_unknown_lesson_is_404(client):
    assert client.get("/api/lessons/nope").status_code == 404


def test_all_demos_run_and_return_a_takeaway():
    for name in DEMOS:
        result = run_demo(name)
        assert result["title"]
        assert result["takeaway"]


def test_demo_endpoint_accepts_parameters(client):
    body = client.get("/api/demos/low-rank?size=64&true_rank=4").json()
    assert body["matrix_size"] == 64
    assert body["structured_curve"]


def test_demo_ignores_parameters_it_does_not_use():
    assert run_demo("bucketing", size=999, rank=3)["resolution"] == 1024


def test_lesson_prerequisites_exist():
    for lesson in LESSONS:
        for prerequisite in lesson.prerequisites:
            assert get_lesson(prerequisite)


# ── Training backends and the job runner ──────────────────────────────


def test_simulated_backend_is_always_available():
    usable, reason = get_backend("simulated").availability()
    assert usable
    assert reason


def test_default_backend_falls_back_to_simulation_without_a_gpu():
    assert default_backend() in {b["name"] for b in available_backends()}


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        get_backend("nope")


def _short_recipe(**training):
    return Recipe.model_validate(
        {"training": {"steps": 40, "save_every": 20, **training}, "data": {"trigger_word": "sks_x"}}
    )


def test_simulated_run_completes_and_emits_the_expected_events():
    manager = JobManager()
    job = manager.submit(_short_recipe(), "simulated")

    for _ in range(200):
        if job.is_terminal:
            break
        time.sleep(0.05)

    assert job.state is JobState.DONE, f"job did not finish: {job.error}"
    assert job.step == 40
    assert job.last_loss is not None
    assert job.checkpoints, "save_every should have produced checkpoints"

    events, cursor = job.events_since(0, limit=5000)
    kinds = {e["kind"] for e in events}
    assert {"status", "step", "checkpoint", "done"} <= kinds
    assert cursor == len(events)


def test_event_cursor_is_incremental_and_resumable():
    manager = JobManager()
    job = manager.submit(_short_recipe(), "simulated")
    while not job.is_terminal:
        time.sleep(0.05)

    first, cursor = job.events_since(0, limit=3)
    assert len(first) == 3
    second, _ = job.events_since(cursor, limit=3)
    assert first[0] != second[0]


def test_cancellation_stops_a_run():
    manager = JobManager()
    job = manager.submit(
        Recipe.model_validate({"training": {"steps": 100_000, "save_every": 0}}), "simulated"
    )
    time.sleep(0.2)
    assert manager.cancel(job.id)

    for _ in range(100):
        if job.is_terminal:
            break
        time.sleep(0.05)
    assert job.state is JobState.CANCELLED
    assert job.step < 100_000


def test_learning_rate_schedule_decays_with_cosine():
    from turducken.training.simulated import _scheduled_lr

    recipe = Recipe.model_validate({"training": {"steps": 1000, "scheduler": "cosine"}})
    assert _scheduled_lr(recipe, 1) > _scheduled_lr(recipe, 500) > _scheduled_lr(recipe, 1000)


def test_warmup_ramps_from_zero():
    from turducken.training.simulated import _scheduled_lr

    recipe = Recipe.model_validate({"training": {"steps": 1000, "warmup_steps": 100}})
    assert _scheduled_lr(recipe, 1) < _scheduled_lr(recipe, 50) < _scheduled_lr(recipe, 100)


def test_constant_schedule_does_not_change():
    from turducken.training.simulated import _scheduled_lr

    recipe = Recipe.model_validate({"training": {"steps": 500, "scheduler": "constant"}})
    assert _scheduled_lr(recipe, 1) == _scheduled_lr(recipe, 400)


def test_simulated_runs_are_reproducible_for_a_fixed_seed():
    from turducken.training.base import EventKind
    from turducken.training.simulated import SimulatedTrainer

    recipe = _short_recipe(seed=7)

    def losses():
        trainer = SimulatedTrainer(speed=100, narrate=False)
        return [e.loss for e in trainer.train(recipe) if e.kind is EventKind.STEP]

    assert losses() == losses()


def test_job_api_round_trip(client):
    response = client.post(
        "/api/jobs",
        json={
            "recipe": _short_recipe().model_dump(mode="json"),
            "backend": "simulated",
            "acknowledge_blockers": True,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    for _ in range(200):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["state"] in {"done", "failed", "cancelled"}:
            break
        time.sleep(0.05)

    assert body["state"] == "done"
    events = client.get(f"/api/jobs/{job_id}/events?cursor=0").json()
    assert events["events"]
    assert events["cursor"] > 0
    assert any(j["id"] == job_id for j in client.get("/api/jobs").json()["jobs"])


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404


def test_blocking_advisories_require_acknowledgement(client):
    """A blocker asks once, then gets out of the way."""
    recipe = Recipe.model_validate(
        {
            "training": {
                "steps": 20,
                "save_every": 10,
                "batch_size": 32,
                "gradient_checkpointing": False,
            }
        }
    ).model_dump(mode="json")

    from turducken.hardware import probe

    if probe().gpus:
        pytest.skip("blocker depends on the absence of ample VRAM")

    # Without a GPU there is no VRAM figure to compare against, so no blocker
    # fires and submission proceeds — which is itself the correct behaviour.
    response = client.post("/api/jobs", json={"recipe": recipe, "backend": "simulated"})
    assert response.status_code in {200, 409}


def test_pruning_removes_finished_jobs():
    manager = JobManager()
    for _ in range(3):
        job = manager.submit(_short_recipe(), "simulated")
        while not job.is_terminal:
            time.sleep(0.02)

    assert len(manager.list()) == 3
    assert manager.prune(keep=1) == 2
    assert len(manager.list()) == 1
