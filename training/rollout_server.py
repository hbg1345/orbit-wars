"""Rollout server (the ACTOR). Run this on the FAST-AT-GAMES machine (the Mac).

It holds a persistent pool of game workers (the same _worker_init / _rollout_one
used by train_rl_v2's local pool) and serves rollout batches over a socket: receive
a list of tasks, run them across the workers, send the results back in order.

The learner (train_rl_v2.py --remote HOST:PORT) on the PC drives it: every iter it
ships that iter's tasks here, the games run on this machine, and the trajectories go
back to the PC for the PPO update.

Run (Mac, conda env `orbit`):
    python training/rollout_server.py --port 5555 --workers 6
Then on the PC:
    python training/train_rl_v2.py --remote <mac-tailscale-ip>:5555 ...
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # train_rl_v2, rollout_net
sys.path.insert(0, os.path.dirname(_HERE))      # repo root: features, main, model

import argparse
import socket
import time
from concurrent.futures import ProcessPoolExecutor

import train_rl_v2 as T
from rollout_net import recv_obj, send_obj


def serve_one(conn, addr, pool):
    print(f"client connected: {addr}")
    n_iter = 0
    while True:
        tasks = recv_obj(conn)                  # list of task tuples (or None to quit)
        if tasks is None:
            print("client sent shutdown sentinel")
            return
        t0 = time.perf_counter()
        results = list(pool.map(T._rollout_one, tasks))
        send_obj(conn, results)
        n_iter += 1
        print(f"  served batch {n_iter}: {len(tasks)} games in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (0.0.0.0 = all interfaces, incl. Tailscale)")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--workers", type=int, default=T.WORKERS,
                    help="game worker processes (match this machine's core count)")
    args = ap.parse_args()

    pool = ProcessPoolExecutor(max_workers=max(1, args.workers),
                               initializer=T._worker_init)
    # warm the workers so the first real batch isn't paying import/spawn cost
    list(pool.map(int, range(args.workers)))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"rollout server up on {args.host}:{args.port} with {args.workers} workers")
    print("waiting for the learner (train_rl_v2.py --remote ...) to connect")

    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                serve_one(conn, addr, pool)
            except (ConnectionError, EOFError) as e:
                # The learner died/restarted. Keep the worker pool warm and wait for
                # it to reconnect instead of tearing everything down.
                print(f"client disconnected ({e}); waiting for a new connection")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        srv.close()
        pool.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()
