import torch
from ultralytics import YOLO

print(f"CUDA is available: {torch.cuda.is_available()}")
model = YOLO('models/best.pt')  

results = model.predict('./input_file/08fd33_4.mp4', save=True)
print(results[0])
print("===============================")
for box in results[0].boxes:
    print(box)