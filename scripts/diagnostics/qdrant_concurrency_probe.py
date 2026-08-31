"""Read-only concurrency probe for the embedded Qdrant store."""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QDRANT_PATH = PROJECT_ROOT / "output" / "qdrant_storage"
COLLECTION_NAME = "rag_rules_bge_small_zh_v1_5"
CONCURRENCY_TIMEOUT_SECONDS = 5.0


def _failure(exception: BaseException, duration_ms: float) -> dict[str, Any]:
    return {
        "status": "failure",
        "exception_type": type(exception).__name__,
        "error": str(exception).replace("\n", " "),
        "duration_ms": round(duration_ms, 3),
    }


def _read_once() -> dict[str, Any]:
    started = time.perf_counter()
    client = None
    try:
        client = QdrantClient(path=str(QDRANT_PATH))
        collection = client.get_collection(COLLECTION_NAME)
        point_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count
        return {
            "status": "success",
            "exception_type": "none",
            "point_count": int(point_count),
            "collection_status": str(collection.status),
            "duration_ms": round(
                (time.perf_counter() - started) * 1000,
                3,
            ),
        }
    except Exception as exc:
        return _failure(
            exc,
            (time.perf_counter() - started) * 1000,
        )
    finally:
        if client is not None:
            client.close()


def _timeout_result() -> dict[str, Any]:
    return {
        "status": "failure",
        "exception_type": "TimeoutError",
        "error": (
            f"operation exceeded {CONCURRENCY_TIMEOUT_SECONDS:.1f}s timeout"
        ),
        "duration_ms": round(CONCURRENCY_TIMEOUT_SECONDS * 1000, 3),
    }


def _scenario(name: str, started: float, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "status": (
            "success"
            if results and all(result["status"] == "success" for result in results)
            else "failure"
        ),
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "results": results,
    }


def run_sequential() -> dict[str, Any]:
    started = time.perf_counter()
    results = [_read_once(), _read_once()]
    return _scenario("sequential_open_read_close", started, results)


def _thread_worker(
    index: int,
    barrier: threading.Barrier,
    results: list[dict[str, Any] | None],
) -> None:
    try:
        barrier.wait(timeout=CONCURRENCY_TIMEOUT_SECONDS)
        results[index] = _read_once()
    except Exception as exc:
        results[index] = _failure(exc, 0.0)


def run_threads() -> dict[str, Any]:
    started = time.perf_counter()
    barrier = threading.Barrier(2)
    results: list[dict[str, Any] | None] = [None, None]
    threads = [
        threading.Thread(
            target=_thread_worker,
            args=(index, barrier, results),
            name=f"qdrant-probe-thread-{index + 1}",
            daemon=True,
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()

    deadline = time.perf_counter() + CONCURRENCY_TIMEOUT_SECONDS
    for thread in threads:
        thread.join(max(0.0, deadline - time.perf_counter()))

    final_results = []
    for index, thread in enumerate(threads):
        if thread.is_alive():
            final_results.append(_timeout_result())
        elif results[index] is None:
            final_results.append(
                _failure(RuntimeError("thread returned no result"), 0.0)
            )
        else:
            final_results.append(results[index])
    return _scenario("two_threads_open_read_close", started, final_results)


def _process_worker(
    index: int,
    send_conn: Any,
    start_event: Any,
) -> None:
    try:
        send_conn.send(("ready", index))
        if not start_event.wait(CONCURRENCY_TIMEOUT_SECONDS):
            send_conn.send(("result", _timeout_result()))
            return
        send_conn.send(("result", _read_once()))
    except Exception as exc:
        try:
            send_conn.send(("result", _failure(exc, 0.0)))
        except Exception:
            pass
    finally:
        send_conn.close()


def _drain_process_messages(
    receive_conns: list[Any],
    messages: list[list[tuple[str, Any]]],
) -> None:
    for index, receive_conn in enumerate(receive_conns):
        while True:
            try:
                if not receive_conn.poll(0):
                    break
                messages[index].append(receive_conn.recv())
            except (EOFError, OSError):
                break


def run_processes() -> dict[str, Any]:
    started = time.perf_counter()
    context = mp.get_context("spawn")
    start_event = context.Event()
    receive_conns = []
    send_conns = []
    processes = []
    messages: list[list[tuple[str, Any]]] = [[], []]

    for index in range(2):
        receive_conn, send_conn = context.Pipe(duplex=False)
        receive_conns.append(receive_conn)
        send_conns.append(send_conn)
        processes.append(
            context.Process(
                target=_process_worker,
                args=(index, send_conn, start_event),
                name=f"qdrant-probe-process-{index + 1}",
                daemon=True,
            )
        )

    try:
        for process in processes:
            process.start()
        for send_conn in send_conns:
            send_conn.close()

        ready = set()
        ready_deadline = time.perf_counter() + CONCURRENCY_TIMEOUT_SECONDS
        while len(ready) < 2 and time.perf_counter() < ready_deadline:
            _drain_process_messages(receive_conns, messages)
            for index, process_messages in enumerate(messages):
                if any(
                    kind == "ready" and value == index
                    for kind, value in process_messages
                ):
                    ready.add(index)
            if len(ready) < 2:
                time.sleep(0.01)

        start_event.set()

        join_deadline = time.perf_counter() + CONCURRENCY_TIMEOUT_SECONDS
        for process in processes:
            process.join(max(0.0, join_deadline - time.perf_counter()))
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(1.0)
        _drain_process_messages(receive_conns, messages)
    finally:
        start_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(1.0)
        for send_conn in send_conns:
            try:
                send_conn.close()
            except Exception:
                pass
        for receive_conn in receive_conns:
            receive_conn.close()

    results = []
    for index, process_messages in enumerate(messages):
        result_messages = [
            value for kind, value in process_messages if kind == "result"
        ]
        if result_messages:
            results.append(result_messages[-1])
        elif processes[index].is_alive():
            results.append(_timeout_result())
        elif processes[index].exitcode not in (0, None):
            results.append(
                _failure(
                    RuntimeError(
                        f"child process exited with code {processes[index].exitcode}"
                    ),
                    0.0,
                )
            )
        else:
            results.append(
                _failure(RuntimeError("process returned no result"), 0.0)
            )
    return _scenario("two_processes_open_read_close", started, results)


def _print_scenario(scenario: dict[str, Any]) -> None:
    print(
        f"{scenario['name']}: {scenario['status']} "
        f"duration_ms={scenario['duration_ms']:.3f}"
    )
    for index, result in enumerate(scenario["results"], start=1):
        print(
            f"  read_{index}: {result['status']} "
            f"exception={result['exception_type']} "
            f"duration_ms={result['duration_ms']:.3f}"
        )
        if result.get("error"):
            print(f"    error={result['error']}")


def main() -> int:
    if not QDRANT_PATH.is_dir():
        print(f"storage_missing: {QDRANT_PATH}")
        return 2

    print("P0-CONC-MIN: read-only Embedded Qdrant concurrency probe")
    print(f"storage={QDRANT_PATH}")
    print(f"collection={COLLECTION_NAME}")
    print("operations=get_collection,count(exact=True)")
    scenarios = [run_sequential(), run_threads(), run_processes()]
    for scenario in scenarios:
        _print_scenario(scenario)
    baseline_ok = scenarios[0]["status"] == "success"
    print(
        "baseline="
        + ("valid" if baseline_ok else "invalid")
    )
    return 0 if baseline_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
