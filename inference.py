# import torch
# import cv2
# import numpy as np
# import os
# from train import HybridEnhancer, apply_hybrid_curves, TARGET_SIZE

# def process_image(model, image_path, device):
#     # 1. Load Image
#     img_orig = cv2.imread(image_path)
#     img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
#     h_orig, w_orig = img_orig.shape[:2]

#     # 2. Proportional Resize (Longest side = TARGET_SIZE)
#     scale = TARGET_SIZE / max(h_orig, w_orig)
#     new_h, new_w = int(h_orig * scale), int(w_orig * scale)
#     img_res = cv2.resize(img_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)

#     # 3. Pad to Square (Matching training behavior)
#     pad_h = max(0, TARGET_SIZE - new_h)
#     pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
#     pad_w = max(0, TARGET_SIZE - new_w)
#     pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
#     img_padded = cv2.copyMakeBorder(img_res, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

#     # 4. Prepare Tensor
#     img_t = torch.from_numpy(img_padded.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
#     img_t = img_t.to(device)

#     # 5. Model Inference
#     with torch.no_grad():
#         global_p, grid_p = model(img_t)
#         enhanced_t = apply_hybrid_curves(img_t, global_p, grid_p)

#     # 6. Convert back to Numpy
#     enhanced_np = enhanced_t.squeeze(0).cpu().permute(1, 2, 0).numpy()
#     enhanced_np = (enhanced_np * 255).astype(np.uint8)

#     # 7. Remove Padding (Crop back to the resized aspect ratio)
#     # We crop the padding we added in step 3
#     crop_h_end = TARGET_SIZE - pad_bottom
#     crop_w_end = TARGET_SIZE - pad_right
#     enhanced_final = enhanced_np[pad_top:crop_h_end, pad_left:crop_w_end]

#     # 8. Optional: Resize back to original dimensions
#     enhanced_final = cv2.resize(enhanced_final, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
    
#     return cv2.cvtColor(enhanced_final, cv2.COLOR_RGB2BGR)

# def main():
#     # Setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = "models/latest_bilateral_model.pth"
#     input_folder = "DataSet_Resized/Images"
#     output_folder = "Global_and_Bilateral_Results"
#     os.makedirs(output_folder, exist_ok=True)

#     # Load Model
#     model = HybridEnhancer().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     print(f"✅ Model loaded from {model_path}")

#     # Process Folder
#     extensions = (".jpg", ".jpeg", ".png")
#     image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(extensions)]

#     for filename in image_files:
#         print(f"Processing {filename}...")
#         result = process_image(model, os.path.join(input_folder, filename), device)
#         cv2.imwrite(os.path.join(output_folder, filename), result)

#     print(f"✨ Done! Results saved in {output_folder}")

# if __name__ == "__main__":
#     main()

import torch
import cv2
import numpy as np
import os
from train import HybridEnhancer, apply_hybrid_curves, TARGET_SIZE

def process_image_highres(model, image_path, device):
    # 1. Load Original High-Res Image (e.g., 6K)
    img_orig_bgr = cv2.imread(image_path)
    img_orig_rgb = cv2.cvtColor(img_orig_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_orig_rgb.shape[:2]

    # 2. Create the "Analysis" version (1024px) for the model to see
    # We do exactly what training did to get the parameters
    scale = TARGET_SIZE / max(h_orig, w_orig)
    new_h, new_w = int(h_orig * scale), int(w_orig * scale)
    img_analysis = cv2.resize(img_orig_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_h = max(0, TARGET_SIZE - new_h)
    pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
    pad_w = max(0, TARGET_SIZE - new_w)
    pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
    img_padded = cv2.copyMakeBorder(img_analysis, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

    # 3. Predict Parameters (Low-Res pass)
    img_analysis_t = torch.from_numpy(img_padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        global_p, grid_p = model(img_analysis_t)

    # 4. Apply to FULL RESOLUTION (High-Res pass)
    # Convert the 6K image to a tensor. 
    # Note: If you run out of VRAM, we can move this to CPU.
    img_full_t = torch.from_numpy(img_orig_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        # apply_hybrid_curves uses grid_sample which scales internally to the image size!
        enhanced_full_t = apply_hybrid_curves(img_full_t, global_p, grid_p)

    # 5. Finalize
    enhanced_final = (enhanced_full_t.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(enhanced_final, cv2.COLOR_RGB2BGR)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/latest_bilateral_model.pth"
    input_folder = "E:\ApixVisuals\PACT"
    output_folder = "HighRes_Bilateral_Results"
    os.makedirs(output_folder, exist_ok=True)

    model = HybridEnhancer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded. Ready for High-Res processing on {device}.")

    extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(extensions)]

    for filename in image_files:
        print(f"🚀 Processing 6K+ Image: {filename}...")
        try:
            result = process_image_highres(model, os.path.join(input_folder, filename), device)
            cv2.imwrite(os.path.join(output_folder, filename), result)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ GPU Memory Full for {filename}. Image is likely too large for your VRAM.")
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"✨ Done! Results saved in {output_folder}")

if __name__ == "__main__":
    main()