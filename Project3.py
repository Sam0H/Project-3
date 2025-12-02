import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_PATH = 'motherboard_image.JPEG' # Check extension (JPEG vs jpg) based on your file
DATA_YAML_PATH = os.path.join('data', 'data.yaml')
EVAL_DIR = os.path.join('data', 'evaluation')

# ==========================================
# STEP 1: OBJECT MASKING (OpenCV)
# ==========================================
def step1_object_masking():
    print(f"--- Step 1: Processing {IMAGE_PATH} ---")
    
    # 1. Read Image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not find image at {IMAGE_PATH}")
        return

    # 2. Convert to Grayscale & Blur 
    # INCREASE blur size to (9,9) to smooth out the internal components
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # 3. Canny Edge Detection
    # LOWER these numbers significantly to catch the faint board edge
    edges = cv2.Canny(blur, 35, 48) 
    
    # 4. Dilate Edges 
    # INCREASE iterations to 4 or 5 to close the gaps in the border
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    # 5. Find Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Filter Contours (Keep the largest one - the PCB)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a black mask
        mask = np.zeros_like(gray)
        
        # Draw the largest contour in white on the mask (Fill it)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        
        # 7. Extract PCB using Bitwise AND
        result = cv2.bitwise_and(img, img, mask=mask)

        # 8. Visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 3, 2)
        plt.title("Mask")
        plt.imshow(mask, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title("Extracted")
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()

        # Save for your report
        cv2.imwrite("step1_mask.png", mask)
        cv2.imwrite("step1_extracted.png", result)
        print("Step 1 Complete. Images saved. Check if background is BLACK.")
    else:
        print("No contours found!")
        
# ==========================================
# STEP 2: YOLO TRAINING
# ==========================================
def step2_training():
    print("--- Step 2: Starting YOLOv11 Training ---")
    
    # Load the Nano model as requested (yolov11n.pt)
    # Ensure you do NOT use v12 
    model = YOLO('yolo11n.pt') 

    # Train the model
    # INDENTATION FIXED BELOW:
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=150,
        imgsz=900,      
        batch=1,        # <--- Reduced to 1 to fit in 4GB VRAM
        name='pcb_model',   
        device=0            
    )
    print("Step 2 Complete. Check 'runs/detect/pcb_model' for results.")
# ==========================================
# STEP 3: EVALUATION
# ==========================================
def step3_evaluation():
    print("--- Step 3: Evaluation on Test Images ---")
    
    # Load your best trained model
    # Note: The path depends on where training saved it. Usually runs/detect/pcb_model/weights/best.pt
    model_path = os.path.join('runs', 'detect', 'pcb_model9', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Did training finish?")
        # Fallback to pre-trained if local training hasn't run yet for testing logic
        model = YOLO('yolo11n.pt') 
    else:
        model = YOLO(model_path)

    # Get list of images in evaluation folder
    if os.path.exists(EVAL_DIR):
        image_files = [f for f in os.listdir(EVAL_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(EVAL_DIR, img_file)
            
            # Run inference
            # conf=0.25 is standard, adjust if missing components
            results = model.predict(img_path, save=True, imgsz=900, conf=0.25)
            
            # Show results
            for result in results:
                result.show() # Opens a window with the detection
    else:
        print(f"Evaluation directory {EVAL_DIR} not found.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Uncomment the steps
    
    #step1_object_masking()
    
    #step2_training() 
    
    # Run this after training
    step3_evaluation()