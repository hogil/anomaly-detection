"""ft → aw_micro → g40_strategy 자동 실행"""
import subprocess, time, os, json

def wait_for(prefix, name):
    print(f"\n=== Waiting for {name} ===")
    while True:
        dirs = [d for d in os.listdir('logs') if d.startswith(prefix) and os.path.isdir(f'logs/{d}')]
        done = [d for d in dirs if os.path.exists(f'logs/{d}/best_info.json')]
        print(f"{name}: {len(done)}/{len(dirs)}")
        if dirs and len(done) >= len(dirs):
            time.sleep(30)
            new = [d for d in os.listdir('logs') if d.startswith(prefix) and os.path.isdir(f'logs/{d}')]
            if len(new) == len(dirs):
                print(f"{name} stable")
                break
        time.sleep(60)

# 1. ft 끝날 때까지
wait_for('ft_', 'ft')

# 2. aw_micro 실행
print("\n=== Running aw_micro ===")
subprocess.run("python experiments_aw_micro.py 2>&1 | tee logs/aw_micro_log.txt",
               shell=True, cwd="D:/project/anomaly-detection")

# 3. g40_strategy 실행
print("\n=== Running g40_strategy ===")
subprocess.run("python experiments_g40_strategy.py 2>&1 | tee logs/g40_strategy_log.txt",
               shell=True, cwd="D:/project/anomaly-detection")

print("\nAll done!")
