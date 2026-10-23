"""Replace an existing instance of the same local server, without killing other apps."""

import errno
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
from contextlib import contextmanager


def listener_pid(port):
    result = subprocess.run(
        ['lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'],
        capture_output=True, text=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f'Cannot identify the process on port {port}: {result.stderr.strip()}')
    pids = {int(value) for value in result.stdout.split()}
    if len(pids) != 1:
        raise RuntimeError(f'Cannot uniquely identify the existing server on port {port}.')
    return pids.pop()


def is_same_server(pid, script):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                            capture_output=True, text=True, check=True)
    command = result.stdout.strip()
    script = Path(script).resolve()
    if command.endswith(' ' + str(script)):
        return True
    # Support the common relative invocations: python server.py and python analysis/.../server.py.
    cwd_result = subprocess.run(['lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'],
                                capture_output=True, text=True, check=True)
    for line in cwd_result.stdout.splitlines():
        if line.startswith('n'):
            try:
                relative = script.relative_to(Path(line[1:]))
            except ValueError:
                continue
            if command.endswith(' ' + str(relative)) or command.endswith(' ./' + str(relative)):
                return True
    return False


@contextmanager
def replacing_server(server_class, address, handler, script, lock_path, timeout=15):
    """Serialize startup, stop only the matching listener, and bind the new instance."""
    with Path(lock_path).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            server = server_class(address, handler)
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            pid = listener_pid(address[1])
            if pid == os.getpid() or not is_same_server(pid, script):
                raise RuntimeError(f'Port {address[1]} belongs to another application (PID {pid}); left it running.') from error
            print(f'Replacing previous server instance (PID {pid})...', flush=True)
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    server = server_class(address, handler)
                    break
                except OSError as retry_error:
                    if retry_error.errno != errno.EADDRINUSE:
                        raise
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Previous server has not released its port; try again after requests finish.') from retry_error
                    time.sleep(.1)
        try:
            yield server
        except BaseException:
            server.server_close()
            raise
