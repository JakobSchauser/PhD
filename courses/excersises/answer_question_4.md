# Answer to Question 4: Image Upscaling and Quality Enhancement

## Question
Damn, I accidentally ruined the quality of my picture. Upscale it (Appendix C).

## Approach
This task involves using AI-powered super-resolution techniques to enhance and upscale a low-quality or damaged image. Modern deep learning models can significantly improve image quality, resolution, and restore lost details.

## Solution Strategy

### Method 1: Real-ESRGAN for General Purpose Upscaling
```python
from transformers import pipeline
import torch

# Use Real-ESRGAN for high-quality upscaling
upscaler = pipeline(
    "image-super-resolution",
    model="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
)

# Load damaged image
damaged_image = load_image("appendix_c_damaged.jpg")

# Upscale and enhance
enhanced_image = upscaler(damaged_image)

# Save result
enhanced_image.save("restored_image_4x.jpg")
```

### Method 2: ESRGAN Models on Hugging Face
```python
from diffusers import StableDiffusionUpscalePipeline
import torch

# Load Stable Diffusion upscaler
upscaler = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    torch_dtype=torch.float16
)

# Upscale with AI enhancement
prompt = "high quality, detailed, sharp, professional photography"
upscaled = upscaler(
    prompt=prompt,
    image=damaged_image,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

upscaled.save("ai_upscaled_result.jpg")
```

### Method 3: SwinIR for Lightweight Super-Resolution
```python
from transformers import Swin2SRImageProcessor, Swin2SRForImageSuperResolution
import torch
from PIL import Image

# Load SwinIR model
processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-lightweight-x4-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-lightweight-x4-64")

# Process image
inputs = processor(damaged_image, return_tensors="pt")

# Generate high-resolution version
with torch.no_grad():
    outputs = model(**inputs)

# Convert to PIL Image
sr_image = processor.postprocess(outputs.reconstruction, damaged_image)
sr_image.save("swinir_upscaled.jpg")
```

### Method 4: BSRGAN for Real-World Image Restoration
```python
from transformers import pipeline

# BSRGAN is excellent for real-world degraded images
bsrgan_pipeline = pipeline(
    "image-super-resolution",
    model="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
)

# Process the damaged image
restored = bsrgan_pipeline(
    damaged_image,
    num_inference_steps=100
)

restored.save("bsrgan_restored.jpg")
```

### Method 5: CodeFormer for Face Enhancement (if applicable)
```python
# If the image contains faces, use specialized face restoration
from transformers import pipeline

face_restorer = pipeline(
    "image-restoration",
    model="microsoft/DiT-XL-2-256"  # or similar face restoration model
)

# Enhance faces specifically
if contains_faces(damaged_image):
    face_enhanced = face_restorer(damaged_image)
    face_enhanced.save("face_restored.jpg")
```

## Implementation Pipeline

### Step 1: Image Analysis and Preprocessing
```python
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def analyze_image_quality(image_path):
    """Analyze what type of degradation the image has"""
    img = cv2.imread(image_path)
    
    # Check for various quality issues
    issues = {
        'low_resolution': img.shape[0] < 512 or img.shape[1] < 512,
        'blurry': cv2.Laplacian(img, cv2.CV_64F).var() < 100,
        'noisy': np.std(img) > 50,
        'low_contrast': img.max() - img.min() < 100,
        'compression_artifacts': detect_jpeg_artifacts(img)
    }
    
    return issues

def preprocess_image(image, issues):
    """Apply appropriate preprocessing based on detected issues"""
    if issues['noisy']:
        # Denoise first
        image = cv2.bilateralFilter(np.array(image), 9, 75, 75)
        image = Image.fromarray(image)
    
    if issues['low_contrast']:
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
    
    return image
```

### Step 2: Multi-Model Approach
```python
def multi_model_upscaling(image):
    """Use multiple models and combine results"""
    results = {}
    
    # Real-ESRGAN
    esrgan_pipeline = pipeline("image-super-resolution", 
                              model="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr")
    results['esrgan'] = esrgan_pipeline(image)
    
    # SwinIR
    swinir_pipeline = pipeline("image-super-resolution",
                              model="caidas/swin2SR-lightweight-x4-64")
    results['swinir'] = swinir_pipeline(image)
    
    # Stable Diffusion Upscaler
    sd_upscaler = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler"
    )
    results['sd'] = sd_upscaler(
        "high quality photograph",
        image=image
    ).images[0]
    
    return results

def ensemble_upscaling(results):
    """Combine multiple upscaling results"""
    # Convert to arrays
    arrays = [np.array(img) for img in results.values()]
    
    # Weighted average (can be more sophisticated)
    weights = [0.4, 0.3, 0.3]  # Prefer ESRGAN slightly
    combined = np.average(arrays, axis=0, weights=weights)
    
    return Image.fromarray(combined.astype(np.uint8))
```

### Step 3: Post-Processing Enhancement
```python
def post_process_enhancement(image):
    """Apply final enhancements"""
    # Sharpen
    from PIL import ImageFilter
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Color enhancement
    enhancer = ImageEnhance.Color(sharpened)
    color_enhanced = enhancer.enhance(1.2)
    
    # Final brightness/contrast adjustment
    enhancer = ImageEnhance.Brightness(color_enhanced)
    final = enhancer.enhance(1.05)
    
    return final
```

## Advanced Techniques

### Progressive Upscaling
```python
def progressive_upscale(image, target_scale=4):
    """Upscale progressively for better quality"""
    current_image = image
    current_scale = 1
    
    while current_scale < target_scale:
        # Upscale by 2x each step
        next_scale = min(current_scale * 2, target_scale)
        scale_factor = next_scale / current_scale
        
        if scale_factor == 2:
            upscaler = pipeline("image-super-resolution",
                               model="caidas/swin2SR-lightweight-x2-64")
        else:
            upscaler = pipeline("image-super-resolution",
                               model="caidas/swin2SR-lightweight-x4-64")
        
        current_image = upscaler(current_image)
        current_scale = next_scale
    
    return current_image
```

### Quality Assessment
```python
def assess_upscaling_quality(original, upscaled):
    """Assess the quality of upscaling"""
    import cv2
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    # Convert to arrays
    orig_array = np.array(original)
    ups_array = np.array(upscaled)
    
    # Resize original for comparison
    orig_resized = cv2.resize(orig_array, (ups_array.shape[1], ups_array.shape[0]))
    
    # Calculate metrics
    psnr = peak_signal_noise_ratio(orig_resized, ups_array)
    ssim = structural_similarity(orig_resized, ups_array, multichannel=True)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'sharpness': cv2.Laplacian(ups_array, cv2.CV_64F).var()
    }
```

## Expected Results
1. **4x resolution increase** (e.g., 256x256 → 1024x1024)
2. **Enhanced detail recovery** - restoration of lost textures and features
3. **Noise reduction** - cleaner, professional-looking result
4. **Improved sharpness** - crisp edges and clear details
5. **Color enhancement** - vibrant, natural colors

## Model Recommendations by Use Case
- **General photos**: Real-ESRGAN, SwinIR
- **Faces/portraits**: CodeFormer, GFPGAN
- **Anime/artwork**: Real-ESRGAN (anime model)
- **Text/documents**: SwinIR classical models
- **Old/damaged photos**: BSRGAN, Old Photo Restoration

## Tools and Models Used
- `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` - Real-world image enhancement
- `stabilityai/stable-diffusion-x4-upscaler` - AI-powered upscaling
- `caidas/swin2SR-lightweight-x4-64` - Efficient super-resolution
- `microsoft/DiT-XL-2-256` - Advanced image restoration
- OpenCV for preprocessing and analysis

## Quality Metrics
- **PSNR** (Peak Signal-to-Noise Ratio) - measures reconstruction quality
- **SSIM** (Structural Similarity Index) - perceptual quality
- **LPIPS** (Learned Perceptual Image Patch Similarity) - deep feature similarity
- **Visual inspection** - human evaluation of naturalness

*Note: The choice of upscaling method depends on the specific type of damage or quality loss in the original image. Multiple approaches may be combined for optimal results.*
