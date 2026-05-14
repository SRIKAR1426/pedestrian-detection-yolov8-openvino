from ultralytics import YOLO
import time
import cv2
import numpy as np
import os

def main():
    print("="*60)
    print("  PEDESTRIAN DETECTION - ACCURATE PIPELINE BENCHMARK")
    print("  YOLOv8s PyTorch vs YOLOv8s OpenVINO INT8")
    print("="*60)

    PT_MODEL = os.path.join("models", "best.pt")
    
    # Try INT8 first, then FP16
    OV_MODEL = os.path.join("models", "openvino_int8", "best_int8_openvino_model")
    if not os.path.exists(OV_MODEL):
        OV_MODEL = os.path.join("models", "openvino_model")
        
    TEST_IMG = os.path.join("test_images", "test.jpg")
    
    # Ensure test image exists or create a dummy one
    os.makedirs("test_images", exist_ok=True)
    if not os.path.exists(TEST_IMG):
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        cv2.imwrite(TEST_IMG, img)

    if not os.path.exists(PT_MODEL):
         print(f"Error: {PT_MODEL} not found.")
         return

    if not os.path.exists(OV_MODEL):
         print(f"Error: {OV_MODEL} not found. Did you run export_openvino.py?")
         return

    # 1. PyTorch Benchmark
    print("\n[1] Warming up PyTorch model...")
    pt_model = YOLO(PT_MODEL, task='detect')
    
    # Warmup
    for _ in range(3):
        pt_model.predict(source=TEST_IMG, verbose=False)

    print("    Running PyTorch Benchmark (30 iterations)...")
    pt_times = []
    
    img_data = cv2.imread(TEST_IMG)

    for i in range(30):
        start = time.time()
        pt_model.predict(source=img_data, verbose=False) # In-memory to avoid disk IO bottleneck
        pt_times.append((time.time() - start) * 1000)

    pt_avg = sum(pt_times) / len(pt_times)
    pt_fps = 1000 / pt_avg
    print(f"    PyTorch Result: {pt_avg:.1f}ms per frame | {pt_fps:.1f} FPS")

    # 2. OpenVINO Benchmark
    print(f"\n[2] Warming up OpenVINO model loaded from {OV_MODEL}...")
    # This natively handles OpenVINO Core compilation. It strictly uses the CPU.
    ov_model = YOLO(OV_MODEL, task='detect')

    # Warmup
    for _ in range(3):
        ov_model.predict(source=img_data, verbose=False)

    print("    Running OpenVINO Benchmark (30 iterations)...")
    ov_times = []
    for i in range(30):
        start = time.time()
        ov_model.predict(source=img_data, verbose=False)
        ov_times.append((time.time() - start) * 1000)

    ov_avg = sum(ov_times) / len(ov_times)
    ov_fps = 1000 / ov_avg
    print(f"    OpenVINO Result: {ov_avg:.1f}ms per frame | {ov_fps:.1f} FPS")

    # 3. Final Summary
    speedup = pt_avg / ov_avg

    print("\n" + "="*50)
    print("         FINAL RESULTS (APPLES-TO-APPLES)")
    print("="*50)
    print("  Note: This measures the FULL pipeline (Pre-processing -> AI -> NMS Post-processing)")
    print(f"  PyTorch model  : {pt_avg:.1f}ms  ({pt_fps:.1f} FPS)")
    print(f"  OpenVINO model : {ov_avg:.1f}ms  ({ov_fps:.1f} FPS)")
    print(f"  Speedup        : {speedup:.2f}x faster with OpenVINO")
    print(f"  Time saved     : {pt_avg - ov_avg:.1f}ms per image")
    print("="*50)
    print("\nSave these benchmark numbers for your project report!")

if __name__ == "__main__":
    main()