import subprocess
import time
import itertools
import json

# Parameters
kd_alpha_values = [0.5]  # , 0.7, 0.9
kd_temperature_values = [5]  # 2, 3,
teacher_model = [
    "BinaryLSTM",
    "BinaryLSTMWithAttention",
    "BinaryL3LSTM",
    "BiLSTMWithAttention",
    "BiLSTM2WithAttention"
]

# Concurrency limit (safe default).
MAX_CONCURRENT_TASKS = 5

# Parameter grid
tasks = list(itertools.product(kd_alpha_values, kd_temperature_values, teacher_model))

def get_gpu_memory_used():
    """Query GPU memory usage via nvidia-smi (MB)."""
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            shell=True
        )
        memory_used = int(output.decode("utf-8").strip().split("\n")[0])
        return memory_used
    except Exception as e:
        print(f"Failed to read GPU memory: {e}")
        return 99999  # Fallback: assume GPU is busy

def run_task(kd_alpha, kd_temperature, teacher_name):
    cmd = (
        f"python distil.py --kd_alpha={kd_alpha} "
        f"--kd_temperature={kd_temperature} --teacher_model={teacher_name}"
    )
    print(f"Starting: {cmd}")
    return subprocess.Popen(cmd, shell=True)

if __name__ == "__main__":
    running_processes = []

    for kd_alpha, kd_temperature, teacher_name in tasks:
        while True:
            # Check running process count
            running_processes = [p for p in running_processes if p.poll() is None]

            if len(running_processes) < MAX_CONCURRENT_TASKS:
                proc = run_task(kd_alpha, kd_temperature, teacher_name)
                running_processes.append(proc)
                break
            else:
                print(f"Reached max concurrency {MAX_CONCURRENT_TASKS}, waiting...")
                time.sleep(5)

        # Optional delay to reduce resource contention
        time.sleep(2)

    for proc in running_processes:
        proc.wait()

    print("All runs finished.")
