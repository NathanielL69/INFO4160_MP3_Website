from flask import Flask, render_template, request, flash, redirect, url_for
import urllib.request
from transformers import ViTImageProcessor, ViTForImageClassification # ViTFeatureExtractor
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import warnings
import os


app = Flask(__name__)

Upload_Folder = 'static/images/'
app.secret_key = "info4160-mp3"
app.config['UPLOAD_FOLDER'] = Upload_Folder
warnings.filterwarnings('ignore')

Extensions = set(['png', 'jpg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Extensions

@app.route('/')
def home():
    #landing page
    return render_template('index.html')

@app.route('/yolo', methods=['GET', 'POST'])
def yolo_image():
    if request.form.get('home_button') == 'Return to Home':
        return render_template('index.html')
    elif request.form.get('yolo_button') == 'Yolo Model':
        return render_template('yolo.html')
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        model = YOLO('yolov8n.pt')
        results = model(Upload_Folder + filename,verbose=False)  # results list
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save(Upload_Folder + 'asian_child.jpg')  # save image
            #filename = im
        
        return render_template('yolo.html', filename=filename)
    else:
        return redirect(request.url)
    
@app.route('/display/<filename>')
def display_img(filename):
    return redirect(url_for('static', filename='images/' + filename), code=301)
    
    

@app.route('/ViT',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if request.form.get('home_button') == 'Return to Home':
            return render_template('index.html')
        elif request.form.get('ViT_button') == 'ViT Model':
            return render_template('ViT.html')
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        img = Upload_Folder + filename
        image = Image.open(img)
        
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        pred_class = model.config.id2label[predicted_class_idx]
        return render_template('ViT.html', pred_class=pred_class)

        
    return render_template('ViT.html')

app.run(debug=False)
