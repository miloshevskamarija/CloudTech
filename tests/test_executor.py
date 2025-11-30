import pytest
pytest.importorskip("dask")
from src.dask_executor import executor


def test_create_local_client_uses_localcluster_and_client(monkeypatch):
    created = {}

    class DummyCluster:
        def __init__(self, n_workers, threads_per_worker, processes):
            created["n_workers"] = n_workers
            created["threads_per_worker"] = threads_per_worker
            created["processes"] = processes

    class DummyClient:
        def __init__(self, cluster):
            created["cluster"] = cluster

    monkeypatch.setattr(executor, "LocalCluster", DummyCluster)
    monkeypatch.setattr(executor, "Client", DummyClient)

    client = executor.create_local_client(n_workers=2, threads_per_worker=3)

    assert isinstance(client, DummyClient)
    assert created["n_workers"] == 2
    assert created["threads_per_worker"] == 3
    assert created["processes"] is False


def test_map_files_creates_delayed_tasks_and_calls_function():
    # The real Dask delayed object has a .compute() method
    filenames = ["file1.root", "file2.root"]
    config = {"answer": 42}

    def process_function(fname, cfg):
        # Stand-in for the real per-file analysis
        return fname, cfg["answer"]

    tasks = executor.map_files(
        client=None,
        filenames=filenames,
        process_function=process_function,
        config=config,
    )

    assert len(tasks) == len(filenames)

    # Compute a couple of delayed results to make sure the graph is correct
    results = [task.compute(scheduler="synchronous") for task in tasks]
    assert results == [("file1.root", 42), ("file2.root", 42)]
