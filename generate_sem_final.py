import numpy as np
import cv2
import os
import random

proc_path = os.path.expanduser("~/Workspace/Hackathons/Wafer_Inspection/data/processed")

def create_sem_base():
    img = np.full((256, 256), 100, dtype=np.uint8)
    noise = np.random.normal(0, 15, (256, 256)).astype(np.uint8)
    return cv2.add(img, noise)

def generate_sem_data():
    classes = {"Particle": 1000, "Pin_Hole": 1000, "Line_Collapse": 1000}
    for label, count in classes.items():
        print(f"Generating {count} distinct SEM samples for {label}...")
        os.makedirs(os.path.join(proc_path, label), exist_ok=True)
        for i in range(count):
            sem_img = create_sem_base()
            if label == "Particle":
                px, py = random.randint(60, 190), random.randint(60, 190)
                cv2.circle(sem_img, (px, py), random.randint(4, 10), 250, -1)
                sem_img = cv2.GaussianBlur(sem_img, (3, 3), 0)
            elif label == "Pin_Hole":
                px, py = random.randint(60, 190), random.randint(60, 190)
                cv2.circle(sem_img, (px, py), random.randint(5, 12), 20, -1)
            elif label == "Line_Collapse":
                for x in range(60, 200, 30):
                    pts = np.array([[x+random.randint(-2,2), 40], [x+random.randint(-5,5), 128], [x+random.randint(-2,2), 210]], np.int32)
                    cv2.polylines(sem_img, [pts], False, 180, 5)
                    if x == 120:
                        cv2.line(sem_img, (x, 128), (x+25, 140), 180, 5)
            cv2.imwrite(os.path.join(proc_path, label, f"SEM_{label}_{i}.png"), sem_img)

if __name__ == "__main__":
    generate_sem_data()
