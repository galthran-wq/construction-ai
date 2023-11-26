# Importing required libs
from flask import Flask, render_template, request
import numpy as np

import albumentations as A
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt

from visualize_eval import eval

CONFIG = {
    "cloud": {
        "path": "./dpt_cloud/checkpoint-10180/",
        "h": 256,
        "w": 256,
        "processor": "Intel/dpt-large-ade",
    },
}


# Preparing and pre-processing the image
def preprocess_img(img_path, model="cloud"):
    preprocessor = CONFIG[model]['processor']
    h = CONFIG[model]['h']
    w = CONFIG[model]['w']
    model_path = CONFIG[model]['path']
    
    transform = A.Compose([
        A.Resize(width=w, height=h),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ])
    processor = AutoImageProcessor.from_pretrained(
        preprocessor, do_resize=False, do_rescale=False, do_normalize=False, ignore_index=0
    )
    img = Image.open(img_path)
    op_img = np.array(img) 
    op_img = op_img / np.iinfo(op_img.dtype).max
    transformed = transform(image=op_img)
    image = transformed['image']
    inputs = processor(image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'][0]
    result = eval(model_path, inputs, with_labels=False)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(img)
    axs[1].imshow(result)
    output = "./static/prediction.png"
    output = "/mnt/sdb1/home/l.morozov/construction-ai/static/prediction2.png"
    fig.savefig(output)
    return output
    # img_resize = op_img.resize((224, 224))
    # img2arr = img_to_array(img_resize) / 255.0
    # return img_reshape

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            preprocess_img(request.files['file'].stream)
            return render_template("result.html")

    except Exception as e:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001, debug=True)
