from runcode_engine import TaskMananger 
import os

if __name__ == "__main__":
    print("start run", __file__)

    task_manager = TaskMananger.TaskMananger(
        "path/to/run_scripts.db",
        "./rand_log",
    )

    # filt = TaskMananger.TaskFilterPipe(
    #     # TaskMananger.TaskFilterParams("lp_norm", "Linf"),
    #     TaskMananger.TaskFilterParams("attack_version", "rand"),
    # )

    task_manager.main_loop(None, cwd=os.path.join(os.getcwd(), "src"))





