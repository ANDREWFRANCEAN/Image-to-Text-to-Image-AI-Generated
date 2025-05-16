# NeuralRender - OCR to AI Image Synthesis

This project combines real-time webcam OCR (Optical Character Recognition) with AI image generation using the [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion) pipeline.

With this tool, you can:
- Capture text from your camera
- Extract that text using Tesseract OCR
- Send the extracted text to a diffusion model to generate a photorealistic image

#Features

- Real-time webcam feed with Tkinter GUI
- Text recognition using `pytesseract`
- Prompt builder system to collect multiple OCR outputs
- AI image generation via Stable Diffusion using `diffusers` and PyTorch
  
##  Requirements
opencv-python
pytesseract
Pillow
torch
torchvision
diffusers
transformers

#Install Tesseract OCR
You must install Tesseract OCR locally:

Download here: https://github.com/tesseract-ocr/tesseract

After installing, update the following line in your script:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# How to Run
Activate your virtual environment:

python -m venv venv
venv\Scripts\activate # On Windows
Install dependencies:

pip install -r requirements.txt
Run the script:

python app.py
Use the GUI:

-Webcam preview appears
-Click Trigger OCR to extract text
-Click Add to Prompts to save text as a prompt
-Click Start AI Processing to quit the GUI and generate images using Stable Diffusion
