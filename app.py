import random
import string

from flask import Flask, render_template, request
import numpy as np
import albumentations as A
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualize_eval import eval
from dataset import FloodNet, CloudDataset, LoveDADataset

CONFIG = {
    "cloud": {
        "path": "./models/dpt_cloud/checkpoint-10180/",
        "h": 256,
        "w": 256,
        "processor": "Intel/dpt-large-ade",
        "to_scale": False,
        "id2class": CloudDataset.ID2CLASS,
    },
    # "floodnet": {
    #     "path": "./mobilenet_floodnet/checkpoint-6351/",
    #     "h": 512,
    #     "w": 512,
    #     "processor": "apple/deeplabv3-mobilevit-small",
    #     "to_scale": False,
    # },
    # "loveda": {
    #     "path": "./mobilenet_loveda/checkpoint-2540/",
    #     "h": 512,
    #     "w": 512,
    #     "processor": "apple/deeplabv3-mobilevit-small",
    #     "to_scale": False,
    # },
    "floodnet": {
        "path": "./models/segformer_floodnet/checkpoint-8760/",
        "h": 256,
        "w": 256,
        "processor": "nvidia/mit-b0",
        "to_scale": True,
        "id2class": FloodNet.ID2CLASS,
    },
    # "loveda": {
    #     "path": "./mobilenet_loveda/checkpoint-2540/",
    #     "h": 512,
    #     "w": 512,
    #     "processor": "apple/deeplabv3-mobilevit-small",
    #     "to_scale": False,
    # },
    "loveda": {
        "path": "./models/beit_loveda/checkpoint-3683/",
        "h": 224,
        "w": 224,
        "processor": "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "to_scale": True,
        "id2class": LoveDADataset.ID2CLASS,
    },
}


# Preparing and pre-processing the image
def preprocess_img(img_path, model="cloud"):
    preprocessor = CONFIG[model]['processor']
    h = CONFIG[model]['h']
    w = CONFIG[model]['w']
    to_scale = CONFIG[model]['to_scale']
    model_path = CONFIG[model]['path']
    ID2CLASS = CONFIG[model]['id2class']
    
    transform = A.Compose([
        A.Resize(width=w, height=h),
        A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ])
    processor = AutoImageProcessor.from_pretrained(
        preprocessor, do_resize=False, do_rescale=False, do_normalize=False, ignore_index=0
    )
    img = Image.open(img_path)
    op_img = np.array(img) 
    if to_scale:
        op_img = op_img / np.iinfo(op_img.dtype).max
    transformed = transform(image=op_img)
    image = transformed['image']
    inputs = processor(image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'][0]
    result = eval(model_path, inputs, with_labels=False, upsampled_h=h, upsampled_w=w)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    # transformed_image = Image.fromarray((transformed['image']*255).astype(np.uint8), 'RGB')
    transformed_image = img
    axs[0].imshow(transformed_image)
    im = axs[1].imshow(result)
    values = ID2CLASS
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) ) for i in range(len(values)) ]
    axs[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
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
        print(e)
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001, debug=True)
