import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process

import psutil


class PredictionProcessingError(Exception):
    def __init__(self, prediction, error):
        self.prediction = prediction
        self.error = error

    def __str__(self):
        return f"Error for prediction {self.prediction}: {self.error}"


def get_max_workers():
    """
    Returns the maximum number of concurrent workers

    The optimal number of workers ultimately depends on how many resources
    each process will call upon.

    To limit this, update the Dockerfile GRAND_CHALLENGE_MAX_WORKERS
    """

    environ_cpu_limit = os.getenv("GRAND_CHALLENGE_MAX_WORKERS")
    cpu_count = multiprocessing.cpu_count()
    return min(
        [
            int(environ_cpu_limit or cpu_count),
            cpu_count,
        ]
    )


def run_prediction_processing(*, fn, predictions):
    """
    Processes predictions in a separate process.

    This takes child processes into account:
    - if any child process is terminated, all prediction processing will abort
    - after prediction processing is done, all child processes are terminated

    Note that the results are returned in completing order.

    Parameters
    ----------
    fn : function
        Function to execute that will process each prediction

    predictions : list
        List of predictions.

    Returns
    -------
    A list of results
    """
    with Manager() as manager:
        results = manager.list()
        errors = manager.list()

        pool_worker = _start_pool_worker(
            fn=fn,
            predictions=predictions,
            max_workers=get_max_workers(),
            results=results,
            errors=errors,
        )
        try:
            pool_worker.join()
        finally:
            pool_worker.terminate()

        for prediction, e in errors:
            import traceback
            traceback.print_exc() 
            raise PredictionProcessingError(
                prediction=prediction,
                error=e,
            ) from e

        return list(results)


def _start_pool_worker(fn, predictions, max_workers, results, errors):
    process = Process(
        target=_pool_worker,
        name="PredictionProcessing",
        kwargs=dict(
            fn=fn,
            predictions=predictions,
            max_workers=max_workers,
            results=results,
            errors=errors,
        ),
    )
    process.start()

    return process


def _pool_worker(*, fn, predictions, max_workers, results, errors):
    caught_exception = False
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Submit the processing tasks of the predictions
            futures = [
                executor.submit(fn, prediction) for prediction in predictions
            ]
            future_to_predictions = {
                future: item
                for future, item in zip(futures, predictions, strict=True)
            }

            for future in as_completed(future_to_predictions):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append((future_to_predictions[future], e))

                    if not caught_exception:  # Hard stop
                        caught_exception = True

                        executor.shutdown(wait=False, cancel_futures=True)
                        _terminate_child_processes()
        finally:
            # Be aggresive in cleaning up any left-over processes
            _terminate_child_processes()


def _terminate_child_processes():
    process = psutil.Process(os.getpid())
    children = process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass  # Not a problem

    # Wait for processes to terminate
    _, still_alive = psutil.wait_procs(children, timeout=5)

    # Forcefully kill any remaining processes
    for p in still_alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass  # That is fine

    # Finally, prevent zombies by waiting for all child processes
    try:
        os.waitpid(-1, 0)
    except ChildProcessError:
        pass  # No child processes, that if fine
