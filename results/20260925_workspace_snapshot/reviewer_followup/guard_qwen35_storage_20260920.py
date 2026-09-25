"""Operational storage guard; never reads scores or changes frozen study files."""
import fcntl
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

MASTER = Path(__file__).resolve().parent / 'qwen35_fixed_multiscale_20260920'
RESERVE_BYTES = 100 * 1024**3


def write_json(path, value):
    tmp = path.with_name(path.name + '.storage_guard.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main():
    assert MASTER.is_dir()
    with (MASTER / 'storage_guard.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            stat = os.statvfs(MASTER)
            free = stat.f_bavail * stat.f_frsize
            record = dict(at=datetime.now(timezone.utc).isoformat(),
                          pid=os.getpid(), available_bytes=free,
                          reserve_bytes=RESERVE_BYTES, state='watching')
            if (MASTER / 'SCHEDULER_COMPLETE.json').exists():
                record['state'] = 'scheduler_finished'
                write_json(MASTER / 'DISK_GUARD_STATUS.json', record)
                print(json.dumps(record), flush=True)
                return
            if free < RESERVE_BYTES:
                control = MASTER / 'CONTROL.json'
                previous = json.loads(control.read_text()) if control.exists() else {}
                record.update(state='low_space', previous_control=previous.copy())
                if previous.get('pause_admission', False):
                    record['action'] = 'preserved_existing_pause'
                else:
                    previous.update(pause_admission=True, storage_guard_pause=dict(
                        at=record['at'], available_bytes=free,
                        reserve_bytes=RESERVE_BYTES,
                        reason='Preserve space for active workers; no automatic resume.'))
                    write_json(control, previous)
                    record['action'] = 'paused_new_admissions'
                write_json(MASTER / 'DISK_GUARD_TRIGGER.json', record)
                write_json(MASTER / 'DISK_GUARD_STATUS.json', record)
                print(json.dumps(record), flush=True)
                return
            write_json(MASTER / 'DISK_GUARD_STATUS.json', record)
            time.sleep(30)


if __name__ == '__main__':
    main()
