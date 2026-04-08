"""데이터 생성 완료 대기 후 v2 실험 시작"""
import subprocess, time, os

print("Waiting for data generation...")
while True:
    if os.path.exists("data/scenarios.csv"):
        # 이미지도 완성됐는지
        n = 0
        for s in ["train","val","test"]:
            for c in ["normal","mean_shift","standard_deviation","spike","drift","context"]:
                d = f"images/{s}/{c}"
                if os.path.exists(d):
                    n += len([f for f in os.listdir(d) if f.endswith(".png")])
        if n >= 6000:
            print(f"Data ready: {n} images")
            break
    time.sleep(30)

# 기존 v2_* 삭제
print("\nCleaning old v2 results...")
import shutil
for d in os.listdir("logs"):
    if d.startswith("v2_"):
        shutil.rmtree(f"logs/{d}", ignore_errors=True)

# v2 학습 실행
print("\nStarting v2 experiments...")
subprocess.run("python experiments_v2.py 2>&1 | tee logs/v2_log.txt",
               shell=True, cwd="D:/project/anomaly-detection")

print("\nAll done!")
