# Local demo guide

## Try the model

Start the app from the project root with `python -m streamlit run app.py`. Choose **Upload a patch** or **Try a bundled test example**. Bundled examples are real images from the held-out test split. The app processes one image at a time.

The input preview shows the patch and its original dimensions. It is converted to RGB and resized to the model's documented input size. The low-color warning is only a simple heuristic; it cannot verify image type.

## Read the model output

The result shows the **top-scoring dataset label** and all three class scores. These scores compare the dataset labels; they are not calibrated disease probabilities or a diagnosis. The labels describe the training dataset.

## Read a Grad-CAM map

Open **View Grad-CAM heatmap** to compare the model input with its overlay. The overlay-strength control changes only the display, not the model score.

The heatmap highlights regions influencing the selected score. It is not a tumor outline or clinical explanation; artifacts and stain/color patterns may influence it.

## Data handling and limitations

The app processes the image locally and does not save or transmit the uploaded file. Use only supported lung histopathology patches. The bundled dataset has no verified patient or slide identifiers, so reported evaluation is duplicate-group-separated, not patient-independent. Dataset reuse and redistribution rights remain unverified. This is an educational research prototype, not for diagnosis or treatment.
