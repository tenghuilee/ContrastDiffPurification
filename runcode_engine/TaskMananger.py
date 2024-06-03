import concurrent.futures
import copy
import enum
import io
import json
import os
import re
import sqlite3
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch

DEFAULT_SEED = "121 122 123"
DEFAULT_DATA_SEED = "0 1 2 3 4 5 6 7"

__script_path = os.path.abspath(__file__).removesuffix(f"{__name__}.py")


def checkup_seed(seed, data_seed):
    if seed is None:
        seed = DEFAULT_SEED
    if data_seed is None:
        data_seed = DEFAULT_DATA_SEED

    if isinstance(seed, list):
        seed = " ".join(seed)
    if isinstance(data_seed, list):
        data_seed = " ".join(data_seed)
    return seed, data_seed


def hack_script(
    run_script_path: str,
    seed: str,
    data_seed: str,
):
    seed, data_seed = checkup_seed(seed, data_seed)

    try:
        with open(run_script_path, "r") as f:
            run_script_string = f.read()

        # python eval_sde_adv.py
        run_script_string = re.sub(
            r"python (eval_sde_adv[\S]*\.py)",
            f"python {os.path.join(__script_path, 'echo_arg2json.py')} --__run_python_file__ \"\\1\"",
            run_script_string,
        )
        run_script_string = run_script_string\
            .replace("cuda_index=$1", "cuda_index=0")\
            .replace("SEED1=$2", f'SEED1="{seed}"')\
            .replace("SEED2=$3", f'SEED2="{data_seed}"')

        ans_set = set()
        ans_list = []
        ans = os.popen(run_script_string).read().strip()
        for line in ans.split('\n'):
            try:
                item = json.loads(line)  # type: dict
            except Exception as _e:
                print(_e)
                continue
            item["__run_script_path__"] = run_script_path
            if "bpda" not in item["__run_python_file__"]:
                # set default value for non bpda mode
                if 'lp_norm' not in item:
                    item['lp_norm'] = "Linf"

            # remove dumplicated dict
            frozen_dict = frozenset(item.items())
            if frozen_dict not in ans_set:
                ans_set.add(frozen_dict)
                ans_list.append(item)

        # print(f"hack {run_script_path}, tasks {len(ans_list)}")
        return ans_list

    except Exception as e:
        print(e)


def hack_scripts_in_directory(directory: str, seed: str, data_seed: str, filename_filter: callable = None):

    if filename_filter is None:
        filename_filter = lambda x: x.endswith(".sh")

    seed, data_seed = checkup_seed(seed, data_seed)
    ans = []
    for sub_dir in os.listdir(directory):
        # print(sub_dir)
        if not filename_filter(sub_dir):
            continue
        sub_dir = os.path.join(directory, sub_dir)
        ans.extend(hack_script(sub_dir, seed, data_seed))

    return ans


class TaskStatus(enum.Enum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3

@dataclass
class Task:
    id: int = -1
    submit_time: float = -1
    complete_time: float = -1
    status: float = -1
    username: str = ""
    json: dict = None
    json_str: str = field(repr=False, default="")

    def __post_init__(self):
        try:
            if self.json_str == "":
                self.json_str = json.dumps(self.json)

            if self.json is None:
                self.json = json.loads(self.json_str)

        except Exception as e:
            print(e)

    @property
    def exp(self) -> str:
        return self.json.get("exp")

    @property
    def seed(self) -> int:
        return self.json.get("seed")

    @property
    def data_seed(self) -> int:
        return self.json.get("data_seed")

    @property
    def script_path(self) -> str:
        return self.json.get("__run_script_path__")

    @property
    def lp_norm(self) -> str:
        return self.json.get("lp_norm", "Linf")

    @property
    def attack_version(self) -> str:
        return self.json.get("attack_version")

    def __hash__(self) -> int:
        return hash(self.json_str)

    def __eq__(self, __value: object) -> bool:
        return self.json_str == __value.json_str

    def to_bash(self, cuda_index: int) -> str:
        params = copy.deepcopy(self.json)

        __run_python_file__ = params.pop("__run_python_file__")
        __run_script_path__ = params.pop("__run_script_path__")

        out = io.StringIO()
        out.write("#! /bin/bash\n")
        out.write(f"export CUDA_VISIBLE_DEVICES={cuda_index}\n\n")
        out.write(f"python {__run_python_file__} ")
        for k, v in params.items():
            if k == "i":
                # short item
                out.write(f"""-i "{v}" """)
            else:
                out.write(f"""--{k}  "{v}" """)

        return out.getvalue()


class TaskFilter:
    def __init__(self):
        pass

    def __call__(self, task: Task) -> bool:
        return self.is_valid(task)

    def is_valid(self, task: Task) -> bool:
        return False


class TaskFilterParams(TaskFilter):
    def __init__(self, key: str, value: str):
        super().__init__()
        self.key = key
        self.value = value

    def is_valid(self, task: Task) -> bool:
        if self.key not in task.json:
            return False
        return task.json.get(self.key) == self.value


class TaskFilterInScriptPath(TaskFilter):
    def __init__(self, substr: str):
        self.substr = substr
        self.key = "__run_script_path__"

    def is_valid(self, task: Task) -> bool:
        if self.key not in task.json:
            return False
        ans = self.substr in task.json.get(self.key)
        return ans


class TaskFilterPipe(TaskFilter):
    def __init__(self, *filters: list):
        self.filters = filters

    def is_valid(self, task: Task) -> bool:
        for filter in self.filters:
            out = filter(task)
            if out is False:
                return False
        return True


class DataBaseMananger:
    def __init__(self, db_path: str):
        """Not Thread Safe"""
        self.db_path = db_path

        # check lock file
        while self.has_lock_file():
            time.sleep(1)

        # create lock file
        with open(db_path + ".lock", 'a') as f:
            f.write(f"{os.getpid()}")

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table()

        self.tasks = set()  # type: set[Task]

    def __del__(self):
        self.cursor.close()
        self.conn.close()
        # delete lock file
        try:
            os.remove(self.db_path + ".lock")
        except:
            pass

    def has_lock_file(self):
        if os.path.exists(self.db_path + ".lock"):
            # print(f"[pid={os.getpid()}] Database {self.db_path} is locked. Waiting")
            return True
        return False

    def create_table(self):
        # Create the tasks table
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS AllTasks (
            id INTEGER PRIMARY KEY ASC,
            script_path TEXT NOT NULL,
            seed INTEGER NOT NULL,
            data_seed INTEGER NOT NULL,
            submit_time REAL NOT NULL DEFAULT CURRENT_TIMESTAMP,
            complete_time REAL,
            status INTEGER DEFAULT 0,
            username TEXT,
            json_str TEXT UNIQUE NOT NULL
        )""")
        self.conn.commit()

    def submit_tasks(self):
        self.cursor.execute("""SELECT json_str FROM AllTasks""")
        rows = self.cursor.fetchall()
        fetched_tasks = set()
        if rows:
            for row in rows:
                fetched_tasks.add(Task(json=None, json_str=row[0]))
        # pick element in self.tasks but not in fetched_tasks
        new_tasks = self.tasks.difference(fetched_tasks)
        if len(new_tasks) > 0:
            # check dumplicated
            self.cursor.executemany(
                """INSERT INTO AllTasks (script_path, seed, data_seed, json_str) values (?, ?, ?, ?)""",
                [(task.script_path, task.seed, task.data_seed, task.json_str)
                    for task in (self.tasks.difference(fetched_tasks))]
            )
            self.conn.commit()
            self.tasks.clear()  # clear self tasks
            return self.cursor.rowcount
        else:
            return 0

    def load_tasks(self, status: TaskStatus, limit: int = -1, offset: int = 0):
        if limit > 0:
            limit_str = f" LIMIT {int(limit)} OFFSET {int(offset)}"
        else:
            limit_str = ""

        if isinstance(status, list):
            status_str = f"{' OR '.join([f'status = {s.value}' for s in status])}"
        else:
            status_str = f"status = {status.value}"

        self.cursor.execute(
            f"SELECT id, submit_time, complete_time, username, json_str FROM AllTasks WHERE {status_str} ORDER BY data_seed, seed, script_path ASC {limit_str};")
        rows = self.cursor.fetchall()
        if not rows:
            return None

        if len(rows) == 0:
            return None

        return [Task(
            id=row[0],
            submit_time=row[1],
            complete_time=row[2],
            username=row[3],
            json_str=row[4],
            status=status,
        ) for row in rows]

    def append_task(self, task: Task):
        self.tasks.add(task)

    def update_task(
        self,
        task: Task,
        status: TaskStatus,
    ):
        self.cursor.execute(
            f"""UPDATE AllTasks SET status = {status.value}, username = "{os.path.expanduser('~')}", complete_time=datetime() WHERE id = {task.id} """)
        self.conn.commit()
        return self.cursor.lastrowid

    @staticmethod
    def pop_task(db_path: str, task_status=TaskStatus.PENDING, task_filter: TaskFilter = None):
        db = DataBaseMananger(db_path)
        offset = 0
        while True:
            if task_filter is not None:
                tasks = db.load_tasks(task_status, offset=offset, limit=16)
                if tasks is None:
                    break
                offset += 1
                for task in tasks:
                    if task_filter.is_valid(task):
                        db.update_task(task, TaskStatus.RUNNING)
                        return task
            else:
                tasks = db.load_tasks(task_status, offset=offset, limit=1)
                if tasks is None:
                    break
                db.update_task(tasks[0], TaskStatus.RUNNING)
                return tasks[0]

        return None
    
    @staticmethod
    def update_task_once(db_path: str, task: Task, status: TaskStatus):
        db = DataBaseMananger(db_path)
        return db.update_task(task, status)

def _run_one_task(db_path: str, log_dir: str, device: int, task: Task, cwd: str):
    try:
        print(f"[pid={os.getpid()}] [begin] cuda={device}", task.script_path)
        bash = task.to_bash(device)
        log_file = os.path.join(
            log_dir, f"{task.exp}_{task.id}.log")
        with open(log_file, "w") as f:
            process = subprocess.Popen(bash, shell=True, stdout=f, stderr=f, cwd=cwd)
            ret_code = process.wait()
        
        if ret_code == 0:
            # submit run status
            DataBaseMananger.update_task_once(db_path, task, TaskStatus.COMPLETED)
            print(f"[pid={os.getpid()}] [completed] cuda={device}", task.script_path)
        else:
            DataBaseMananger.update_task_once(db_path, task, TaskStatus.FAILED)
            print(f"[pid={os.getpid()}] [failed] cuda={device}, return code={ret_code}", task.script_path)

    except Exception as e:
        DataBaseMananger.update_task_once(db_path, task, TaskStatus.FAILED)
        print(
            f"[pid={os.getpid()}] [failed] cuda={device}\n {e}\n \033[31m{task}\033[0m")

    return device, task


class TaskMananger:
    def __init__(self, db_path: str, log_path: str):
        os.makedirs(log_path, exist_ok=True)
        self.db_path = db_path
        self.log_path = log_path

    def create_tasks_from_script_dir(self, dir_of_scripts: str, seed: str, data_seed: str, filename_filter: callable = None):
        db = DataBaseMananger(self.db_path)
        task_json = hack_scripts_in_directory(
            dir_of_scripts, seed=seed, data_seed=data_seed, filename_filter=filename_filter)
        for task in task_json:
            db.append_task(Task(json=task))

        return db.submit_tasks()
    
    def create_tasks_from_file_list(self, filelist: list[str], seed: str, data_seed: str):
        db = DataBaseMananger(self.db_path)
        for file in filelist:
            tasks_list = hack_script(file, seed=seed, data_seed=data_seed)
            for task in tasks_list:
                db.append_task(Task(json=task))
        return db.submit_tasks()

    def iter_pending_tasks(self, filters: TaskFilter, limit: int):
        db = DataBaseMananger(self.db_path)
        pass_count = 0
        for row in db.load_tasks(status=TaskStatus.PENDING):
            if filters is not None and not filters(row):
                continue
            pass_count += 1
            if pass_count > limit:
                break
            yield row

    def main_loop(self, task_filter: TaskFilter, cwd=None, _num_device=None):

        if _num_device is None:
            num_devices = torch.cuda.device_count()
            if num_devices <= 0:
                raise RuntimeError("No CUDA devices found")
        else:
            num_devices = _num_device
            warnings.warn("_num_device is for debug mode only")
        
        print("[number of devices]", num_devices)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_devices) as executor:

            def _submit_task(device, out: set, _db_path, _log_path, _task_filter, _cwd):
                task = DataBaseMananger.pop_task(
                    _db_path,
                    [TaskStatus.PENDING, TaskStatus.FAILED],
                    _task_filter,
                )
                if task is None:
                    print("No task lefted")
                    return

                fs = executor.submit(_run_one_task, db_path=_db_path,
                                     log_dir=_log_path, device=device, task=task, cwd=_cwd)
                out.add(fs)

            con_futures = set()
            for device in range(num_devices):
                print(f"[pid={os.getpid()}] [submit] cuda={device}")
                _submit_task(device, con_futures, self.db_path, self.log_path, task_filter, cwd)

            while len(con_futures) > 0:
                _finished, _ = concurrent.futures.wait(
                    con_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for cf in _finished:
                    # remove the complete future from the directory
                    _device, _ = cf.result()
                    con_futures.remove(cf)
                    _submit_task(_device, con_futures,
                                 self.db_path, self.log_path, task_filter, cwd)
