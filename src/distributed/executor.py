"""
Dask-based execution helpers

This module hides the details of starting a local Dask cluster
and submitting per-file analysis tasks.
"""

from dask.distributed import Client, LocalCluster
from dask import delayed


def create_local_client(n_workers=4, threads_per_worker=1):
    """
    Create a local Dask client with a LocalCluster.

    Parameters
    ----------
    n_workers : int
        Number of workers to start.
    threads_per_worker : int
        Number of threads per worker.

    Returns
    -------
    dask.distributed.Client
        Connected Dask client.
    """
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=False,  # threads-only, safe in WSL
    )
    client = Client(cluster)
    return client


def map_files(client, filenames, process_function, config):
    """
    Submit a per-file processing function as Dask delayed tasks.

    Parameters
    ----------
    client : dask.distributed.Client
        Active Dask client.
    filenames : list of str
        List of ROOT file paths to process.
    process_function : callable
        Function of the form process_function(filename, config)
        that returns (hist, meta_info).
    config : dict
        Configuration dictionary passed to the processing function.

    Returns
    -------
    list of delayed objects representing per-file results.
    """
    tasks = [delayed(process_function)(filename, config) for filename in filenames]
    return tasks
