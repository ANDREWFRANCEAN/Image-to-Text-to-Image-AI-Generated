import cv2
import pytesseract
from diffusers import StableDiffusionPipeline
import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Specify the path to Tesseract if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize variables
prompts = []

# Open the camera
cam = cv2.VideoCapture(0)
cv2.namedWindow("Text2Image Synthesis Engine ")

# Create Tkinter window
root = tk.Tk()
root.title("Neuralrender")

# Create a label to display the camera feed
label_camera = ttk.Label(root)
label_camera.grid(row=0, column=0, columnspan=2)

# Create a label to display the OCR result
label_ocr_result = ttk.Label(root, text="OCR Result: ")
label_ocr_result.grid(row=1, column=0)

# Create a variable to store the OCR result
ocr_result_var = tk.StringVar()
entry_ocr_result = ttk.Entry(root, textvariable=ocr_result_var, state="readonly")
entry_ocr_result.grid(row=1, column=1)

# Create a button to trigger OCR
def trigger_ocr():
    ret, frame = cam.read()
    if ret:
        cv2.imwrite("Example/capture.png", frame)
        img = cv2.imread('Example/capture.png')
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        ocr_result_var.set(text)
        print(f"OCR Result: {text}")

button_trigger_ocr = ttk.Button(root, text="Trigger OCR", command=trigger_ocr)
button_trigger_ocr.grid(row=2, column=0, columnspan=2)

# Create a button to add the OCR result to prompts
def add_to_prompts():
    text = ocr_result_var.get()
    if text.strip():  # Check if the text is not empty
        prompts.append(text)
        print(f"Added to prompts: {text}")

button_add_to_prompts = ttk.Button(root, text="Add to Prompts", command=add_to_prompts)
button_add_to_prompts.grid(row=3, column=0, columnspan=2)

# Create a button to start AI processing
button_start_processing = ttk.Button(root, text="Start AI Processing", command=root.quit)
button_start_processing.grid(row=4, column=0, columnspan=2)

# Function to update the camera feed
def update_camera_feed():
    ret, frame = cam.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)
        label_camera.img = img
        label_camera.config(image=img)
        label_camera.after(10, update_camera_feed)

update_camera_feed()

root.mainloop()

# Close the camera window
cam.release()
cv2.destroyAllWindows()

# Perform AI processing using the StableDiffusionPipeline
model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Process prompts with the diffusion model
images = []
for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f'picture_{i}.jpg')
    images.append(image)

# Display or save the final processed image
if images:
    final_image = images[-1]
    final_image.show()  # Display the image using the default image viewer
    final_image.save('final_processed_image.jpg')  # Save the image to a file
