"""데이터 생성 → 이미지 생성 → v2 학습 자동 chain"""
import subprocess, os, shutil

# 1. 기존 데이터 삭제
print("=== 1. 기존 데이터 삭제 ===")
for d in ["data", "images", "display"]:
    if os.path.exists(d):
        shutil.rmtree(d)

# 2. 데이터 생성
print("\n=== 2. 데이터 생성 ===")
subprocess.run("python generate_data.py", shell=True)

# 3. 이미지 생성
print("\n=== 3. 이미지 생성 ===")
subprocess.run("python generate_images.py", shell=True)

# 4. 기존 v2 결과 삭제
print("\n=== 4. 기존 v2_* 삭제 ===")
for d in os.listdir("logs"):
    if d.startswith("v2_"):
        shutil.rmtree(f"logs/{d}", ignore_errors=True)

# 5. v2 학습 시작
print("\n=== 5. v2 학습 시작 ===")
subprocess.run("python experiments_v2.py", shell=True)

print("\n=== ALL DONE ===")
