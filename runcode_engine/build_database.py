from TaskMananger import *

OUT_DB_PATH="./target/run_scripts.db"

script_path = [
    "../run_scripts/cifar10",
    # "../run_scripts/cifar100",
    # "../run_scripts/gtsrb",
    # "../run_scripts/imagenet",
    "../run_scripts_contrastive_wideresnet-28-10/cifar10",
    # "../run_scripts_contrastive_wideresnet-28-10/cifar100",
    # "../run_scripts_contrastive_wideresnet-28-10/gtsrb",
    # "../run_scripts_contrastive_wideresnet-28-10/imagenet",
]

def filename_filter(fname: str) -> bool:
    return "square" in fname

if os.path.exists(OUT_DB_PATH):
    os.remove(OUT_DB_PATH)

mananger = TaskMananger(OUT_DB_PATH, "./target")

# ImageNet
# DEFAULT_SEED = "121 122"
# DEFAULT_DATA_SEED = " ".join([str(i) for i in range(32)])

for script in script_path:
    num_lines = mananger.create_tasks_from_script_dir(
        script,
        DEFAULT_SEED,
        DEFAULT_DATA_SEED,
        filename_filter=filename_filter,
    )

    print(f"number of tasks {num_lines} from {script}")

print(f"saved to {OUT_DB_PATH}")

