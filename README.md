ImagetoText: A desktop application that uses computer vision to analyse and describe images in plain English, completely local, no API, 
no internet required after setup.

What it does: 
Upload any image and the app will:
1. Identify the main subject (e.g. a shoe, a purple flower, a cat)
2. Describe colours, lighting, composition, and detail level
3. Show the image preview on the left and the description on the right

Tech Stack:
BLIP (Salesforce) — vision-language model for image captioning
OpenCV – colour analysis, brightness, edge detection
PyTorch — runs the model locally on CPU or GPU
Tkinter — desktop GUI
Pillow - image loading and processing


First run will download the BLIP model (approx 900 MB) and cache it locally.
After that, everything runs offline.

Supported Image Formats: JPG, JPEG, PNG, BMP, WebP, GIF

Requirements:
Python 3.11
macOS / Windows / Linux
Approx 1 GB space for the model cache
