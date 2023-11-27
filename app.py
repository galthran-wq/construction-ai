import random
import string

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
    "floodnet": {
        "path": "./mobilenet_floodnet/checkpoint-6351/",
        "h": 256,
        "w": 256,
        "processor": "apple/deeplabv3-mobilevit-small",
    },
    "loveda": {
        "path": "./mobilenet_loveda/checkpoint-2540/",
        "h": 256,
        "w": 256,
        "processor": "apple/deeplabv3-mobilevit-small",
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
    image_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))
    output = f"static/{image_name}.png"
    print(output)
    fig.savefig(output)
    return output

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
            model = request.form['datasetOptions']
            prediction = preprocess_img(request.files['file'].stream, model=model)
            return render_template("result.html", prediction=prediction)

    except Exception as e:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001, debug=True)
