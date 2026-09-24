import os
import cv2
import numpy as np
import torch
import logging

class ParExtractor:
    def __init__(self, model_path="models/mobilenet_par.pt", device="cpu"):
        self.device = device
        self.model = None
        self.is_onnx = False
        
        # Fallback to .pt if .onnx requested but missing, or vice versa
        if not os.path.exists(model_path):
            alt_path = model_path.replace('.pt', '.onnx') if '.pt' in model_path else model_path.replace('.onnx', '.pt')
            if os.path.exists(alt_path):
                model_path = alt_path
                
        if not os.path.exists(model_path):
            logging.warning(f"PAR model not found at {model_path}. Using dummy PAR extractor.")
            return
            
        try:
            if model_path.endswith('.onnx'):
                import onnxruntime as ort
                self.is_onnx = True
                self.model = ort.InferenceSession(model_path)
            else:
                self.model = torch.jit.load(model_path, map_location=device)
                self.model.eval()
        except Exception as e:
            logging.error(f"Failed to load PAR model: {e}")
            self.model = None

    def preprocess(self, crop):
        # Resize to standard PAR input size (e.g., 256x128 or 224x224 depending on mobilenet)
        # Assuming 256x128 as standard for ReID/PAR
        img = cv2.resize(crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # Normalize (ImageNet stats)
        img -= np.array([0.485, 0.456, 0.406])
        img /= np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0) # Add batch dim
        return img

    def extract(self, crop):
        """
        Extracts Person Attribute Recognition (PAR) features.
        Returns a normalized numpy array of features.
        """
        if crop is None or crop.size == 0:
            return np.zeros(128, dtype=np.float32)
            
        if self.model is None:
            # Fallback: Extract Color Histogram
            # Split crop into top half (shirt) and bottom half (pants)
            h, w = crop.shape[:2]
            top_half = crop[:h//2, :]
            bottom_half = crop[h//2:, :]
            
            # Convert to HSV
            top_hsv = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
            bottom_hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
            
            # Calc Histograms for Hue (0-179) and Saturation (0-255)
            # We use 16 bins for Hue and 8 bins for Saturation = 128 bins per half
            top_hist = cv2.calcHist([top_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            bottom_hist = cv2.calcHist([bottom_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            
            # Normalize
            cv2.normalize(top_hist, top_hist)
            cv2.normalize(bottom_hist, bottom_hist)
            
            features = np.concatenate((top_hist.flatten(), bottom_hist.flatten()))
            return features.astype(np.float32)
            
        input_data = self.preprocess(crop)
        
        if self.is_onnx:
            input_name = self.model.get_inputs()[0].name
            features = self.model.run(None, {input_name: input_data})[0]
        else:
            with torch.no_grad():
                tensor = torch.from_numpy(input_data).to(self.device)
                features = self.model(tensor).cpu().numpy()
                
        # Flatten and L2 normalize
        features = features.flatten()
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm
            
        return features
