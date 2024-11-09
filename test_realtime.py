import cv2
import torch
from DCShadowNet_test import DCShadowNet
import argparse
from utils_loss import RGB2BGR, tensor2numpy, denorm
from PIL import Image
import os
import numpy as np
import time
from model_optimization import optimize_model_for_cpu, prune_model, quantize_model  # 导入优化函数

def parse_args():
    desc = "Real-time shadow removal with DCShadowNet using webcam input"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--ch', type=int, default=64, help='Base channel number per layer')
    parser.add_argument('--dataset', type=str, default='SRD', help='dataset_name')
    parser.add_argument('--samplepath', type=str, default='./samples/', help='dataset_path')
    parser.add_argument('--n_res', type=int, default=4, help='Number of residual blocks')
    parser.add_argument('--img_size', type=int, default=256, help='Size of image for input to the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set device mode [cpu, cuda]')
    parser.add_argument('--model_step', type=int, default=500000, help='Model checkpoint step to load')
    parser.add_argument('--cameraID', default = 0, help = 'Set Camera ID')
    parser.add_argument('--optimize', action='store_true', help='Optimize model for CPU performance')
    parser.add_argument('--prune', action='store_true', help='Prune model parameters')
    parser.add_argument('--quantize', action='store_true', help='Quantize model for optimized performance')
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize DCShadowNet
    gan = DCShadowNet(args)
    gan.build_model()
    gan.load(os.path.join(args.result_dir, args.dataset, 'model'), args.model_step)

    # Check for optimization options
    if args.prune:
        print("Pruning model...")
        gan.genA2B = prune_model(gan.genA2B)

    if args.quantize and args.device == torch.device('cpu'):
        print("Quantizing model for CPU...")
        gan.genA2B = quantize_model(gan.genA2B)

    if args.optimize and args.device == torch.device('cpu'):
        print("Optimizing model for CPU...")
        gan.genA2B = optimize_model_for_cpu(gan.genA2B, img_size=args.img_size)

    gan.genA2B.eval()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit the real-time shadow removal.")

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        real_A = gan.test_transform(img).unsqueeze(0).to(args.device)

        with torch.no_grad():
            fake_A2B, _, _ = gan.genA2B(real_A)

        B_fake = RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))) * 255.0
        B_fake = B_fake.astype(np.uint8)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(B_fake, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Original Image", frame)
        cv2.imshow("Real-time Shadow Removal", B_fake)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
