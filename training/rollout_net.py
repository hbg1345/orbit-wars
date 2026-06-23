"""Tiny length-prefixed socket protocol for the split actor/learner setup.

The PC (learner) runs train_rl_v2.py and, instead of its local ProcessPoolExecutor,
ships each iter's rollout TASKS to a rollout_server.py running on the Mac (actor),
which runs the games and ships the RESULTS back. One message = 8-byte big-endian
length + zlib-compressed pickle. pickle's memo dedups the learner state_dict that is
shared across every task in a batch, so the weights cross the wire ONCE per iter.

No silent failures: a short/closed socket raises, it is not papered over.
"""

import pickle
import socket
import struct
import time
import zlib


def send_obj(sock, obj):
    data = zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    sock.sendall(struct.pack(">Q", len(data)) + data)


def _recv_n(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError(f"socket closed with {len(buf)}/{n} bytes read")
        buf.extend(chunk)
    return bytes(buf)


def recv_obj(sock):
    (n,) = struct.unpack(">Q", _recv_n(sock, 8))
    return pickle.loads(zlib.decompress(_recv_n(sock, n)))


def rollout_connect(host, port, retries=60, wait=2.0):
    """Connect to the rollout server, retrying so the learner can be started before
    (or at the same time as) the actor. Raises if it never comes up."""
    last = None
    for _ in range(retries):
        try:
            s = socket.create_connection((host, port), timeout=30)
            # The 30s timeout was ONLY for establishing the connection. Clear it now:
            # a rollout batch legitimately takes minutes, so recv must block, not time
            # out. (Leaving it set made recv() raise TimeoutError after 30s.)
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"connected to rollout server {host}:{port}")
            return s
        except OSError as e:
            last = e
            print(f"  rollout server {host}:{port} not up yet ({e}); retrying...")
            time.sleep(wait)
    raise ConnectionError(f"could not reach rollout server {host}:{port}: {last}")


def rollout_remote(sock, tasks):
    """Send one iter's tasks, then yield results ONE AT A TIME as the server finishes
    each game (in task order). Streaming (vs one big recv at the end) lets the learner's
    tqdm bar advance per game, so you can watch progress on the PC."""
    tasks = list(tasks)
    send_obj(sock, tasks)
    for _ in range(len(tasks)):
        yield recv_obj(sock)
