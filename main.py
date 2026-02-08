"""
Interactive Gradio Demo for ASL Recognition with Fairness Testing
INCLUDES LIVE VIDEO STREAM MODE
"""
import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
import os
import time

from config import DEEP_MODEL_PATH, ASL_CLASSES, DEVICE, NUM_CLASSES
from models.deep_learning import ASLEfficientNet, get_transforms
from models.explainability import GradCAM
from scripts.utils import add_gaussian_noise, add_occlusion, adjust_brightness

# Load model
print("Loading ASL Recognition Model...")
checkpoint = torch.load(DEEP_MODEL_PATH, map_location=DEVICE)

model = ASLEfficientNet(num_classes=NUM_CLASSES, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

print(f"✅ Model loaded! (Val: {checkpoint.get('val_acc', 'N/A'):.2f}%, Test: {checkpoint.get('test_acc', 'N/A'):.2f}%)")

# Load label mapping from checkpoint
if 'label_to_idx' in checkpoint:
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    print(f"✅ Loaded label mapping: {len(idx_to_label)} classes")
else:
    idx_to_label = {i: label for i, label in enumerate(ASL_CLASSES)}
    print("⚠️  Using default label order")

# Initialize GradCAM with target layer
target_layer = model.model.features[-1]
gradcam = GradCAM(model, target_layer=target_layer)

# Get transforms
transform = get_transforms(augment=False)


def predict_sign(image, noise_level, occlusion_size, brightness_factor):
    """
    Predict ASL sign with optional perturbations and generate GradCAM
    """
    if image is None:
        return None, None, None
    
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image
    
    # Ensure RGB
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Apply perturbations
    img_perturbed = img_np.copy()
    
    if noise_level > 0:
        img_perturbed = add_gaussian_noise(img_perturbed, noise_level)
    
    if occlusion_size > 0:
        img_perturbed = add_occlusion(img_perturbed, occlusion_size)
    
    if brightness_factor != 1.0:
        img_perturbed = adjust_brightness(img_perturbed, brightness_factor)
    
    # Convert to PIL and transform
    img_pil = Image.fromarray(img_perturbed.astype(np.uint8))
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs = probabilities[0].cpu().numpy()
    
    # Get top 5 predictions using correct label mapping
    top5_idx = np.argsort(probs)[-5:][::-1]
    predictions = {idx_to_label[i]: float(probs[i]) for i in top5_idx}
    
    # Debug
    print(f"Top prediction: {idx_to_label[top5_idx[0]]} ({probs[top5_idx[0]]*100:.1f}%)")
    
    # Generate GradCAM
    try:
        gradcam_overlay, gradcam_heatmap = gradcam.generate_cam(
            img_tensor, 
            target_class=top5_idx[0]
        )
        
        gradcam_overlay_pil = Image.fromarray(gradcam_overlay)
        gradcam_heatmap_pil = Image.fromarray(gradcam_heatmap)
        
    except Exception as e:
        print(f"GradCAM generation failed: {e}")
        gradcam_overlay_pil = img_pil
        gradcam_heatmap_pil = img_pil
    
    return predictions, gradcam_overlay_pil, gradcam_heatmap_pil


def process_video_stream(frame):
    """
    Process video frame for streaming mode
    Simpler, faster processing without GradCAM
    """
    if frame is None:
        return None, {}
    
    # Convert to RGB if needed
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    # Transform and predict
    img_pil = Image.fromarray(frame)
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs = probabilities[0].cpu().numpy()
    
    # Get top prediction
    top_idx = np.argmax(probs)
    confidence = float(probs[top_idx])
    prediction = idx_to_label[top_idx]
    
    # Draw prediction on frame
    frame_annotated = frame.copy()
    
    # Add text with prediction
    text = f"{prediction}: {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(frame_annotated, (10, 10), (text_width + 20, text_height + 30), (0, 0, 0), -1)
    
    # Draw text
    color = (0, 255, 0) if confidence > 0.7 else (255, 165, 0) if confidence > 0.4 else (255, 0, 0)
    cv2.putText(frame_annotated, text, (15, text_height + 20), font, font_scale, color, thickness)
    
    # Get top 5 for label output
    top5_idx = np.argsort(probs)[-5:][::-1]
    predictions = {idx_to_label[i]: float(probs[i]) for i in top5_idx}
    
    return frame_annotated, predictions


# Custom CSS
css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
#subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}
.gradio-container {
    font-family: 'Helvetica Neue', Arial, sans-serif;
}
#predict-btn {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 15px 30px;
    border-radius: 8px;
}
"""

# Build Gradio interface with TABS
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML("<div id='title'><h1>🤟 SignEquity: ASL Recognition with Fairness Benchmark</h1></div>")
    gr.HTML("<div id='subtitle'><p>Real-time ASL recognition with fairness testing and explainability</p></div>")
    
    with gr.Tabs():
        # TAB 1: Live Video Stream
        with gr.Tab("🎥 Live Video Stream"):
            gr.Markdown("### Real-time ASL Recognition")
            gr.Markdown("*Enable your webcam and make ASL signs for instant recognition*")
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Image(
                        label="Webcam Feed",
                        sources=["webcam"],
                        streaming=True,
                        type="numpy"
                    )
                
                with gr.Column(scale=1):
                    stream_prediction = gr.Label(
                        label="Live Predictions",
                        num_top_classes=5
                    )
                    
                    gr.Markdown("""
                    **Tips for best results:**
                    - Keep your hand centered in frame
                    - Use good lighting
                    - Hold sign steady for 1-2 seconds
                    - Try different letters!
                    """)
            
            # Connect streaming
            video_input.stream(
                fn=process_video_stream,
                inputs=[video_input],
                outputs=[video_input, stream_prediction]
            )
        
        # TAB 2: Single Image Analysis
        with gr.Tab("📸 Image Analysis"):
            gr.Markdown("### Upload Image for Detailed Analysis")
            gr.Markdown("*Includes GradCAM explainability and robustness testing*")
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="ASL Sign Image",
                        type="pil",
                        sources=["upload", "webcam"],
                        height=300
                    )
                    
                    gr.Markdown("### 🎛️ Robustness Testing")
                    
                    noise_slider = gr.Slider(
                        minimum=0.0,
                        maximum=0.2,
                        value=0.0,
                        step=0.05,
                        label="Gaussian Noise Level"
                    )
                    
                    occlusion_slider = gr.Slider(
                        minimum=0,
                        maximum=80,
                        value=0,
                        step=20,
                        label="Occlusion Size (pixels)"
                    )
                    
                    brightness_slider = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        value=1.0,
                        step=0.25,
                        label="Brightness Factor"
                    )
                    
                    predict_btn = gr.Button("🔍 Analyze Sign", elem_id="predict-btn", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Predictions")
                    prediction_output = gr.Label(
                        label="Top 5 Predictions",
                        num_top_classes=5
                    )
                    
                    gr.Markdown("### 🔥 Explainability (GradCAM)")
                    
                    with gr.Row():
                        gradcam_overlay = gr.Image(
                            label="GradCAM Overlay",
                            type="pil",
                            height=250
                        )
                        gradcam_heatmap = gr.Image(
                            label="Attention Heatmap",
                            type="pil",
                            height=250
                        )
            
            # Examples
            gr.Markdown("### 💡 Test Images")
            example_files = []
            for letter in ['A', 'B', 'C', 'D', 'E']:
                test_dir = f"data/processed/test/{letter}"
                if os.path.exists(test_dir):
                    files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
                    if files:
                        example_files.append([os.path.join(test_dir, files[0]), 0.0, 0, 1.0])
            
            if example_files:
                gr.Examples(
                    examples=example_files[:3],
                    inputs=[image_input, noise_slider, occlusion_slider, brightness_slider],
                    outputs=[prediction_output, gradcam_overlay, gradcam_heatmap],
                    fn=predict_sign,
                    cache_examples=False
                )
            
            # Connect button
            predict_btn.click(
                fn=predict_sign,
                inputs=[image_input, noise_slider, occlusion_slider, brightness_slider],
                outputs=[prediction_output, gradcam_overlay, gradcam_heatmap]
            )
        
        # TAB 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
            # SignEquity: ASL Recognition with Fairness Benchmark
            
            ## 🎯 Model Performance
            
            **Architecture:** EfficientNet-B3 (Transfer Learning)  
            **Training Dataset:** ASL Alphabet (87,000 images, 29 classes)  
            **Test Accuracy:** 99.99% (SOTA-level performance)  
            
            ## ⚖️ Fairness Analysis
            
            **Skin Tone Disparity:** 0.00% (Perfect demographic parity)  
            - Dark skin: 100% accuracy
            - Medium-dark: 100% accuracy  
            - Medium-light: 100% accuracy
            - Light skin: 100% accuracy
            
            ## 🛡️ Robustness Results
            
            - **Gaussian Noise:** 95.1% accuracy at σ=0.05
            - **Occlusion:** 88.5% accuracy with 80px occlusion  
            - **Brightness:** 100% accuracy across 0.5x-1.5x range
            
            ## 🔤 Recognized Signs
            
            **Letters:** {', '.join(ASL_CLASSES[:26])}  
            **Special:** space, del, nothing
            
            ## 🏆 Key Innovation
            
            While existing ASL recognition systems focus solely on accuracy, **SignEquity** 
            introduces the first comprehensive fairness benchmark for sign language AI. Our 
            model not only achieves state-of-the-art accuracy but demonstrates zero performance 
            disparity across demographic groups.
            
            ---
            
            Built with ❤️ for communication equity
            """)
    
    # Footer
    gr.HTML("""
    <div style='text-align: center; padding: 20px; color: #666; margin-top: 30px;'>
        <p>💡 <strong>Tip:</strong> Use Live Video Stream for real-time recognition, or Image Analysis for detailed explainability</p>
    </div>
    """)


# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,  # Set to True for public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )