from __future__ import division, print_function
from io import BytesIO
import base64
from keras.optimizers import Adam
import keras.metrics as t
from skimage import measure
from PIL import Image, ImageDraw
import numpy as np
from keras.models import model_from_json
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
from keras import backend as K
import segmentation_models as sm
import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from wtforms import StringField, SubmitField, SelectField, DateField, FileField, TextAreaField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
import mysql.connector
import hashlib
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
# Define a flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:A_159357456_y@localhost/flaskdb'
app.config['SECRET_KEY'] = 'my-secret-key'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

#call it into front
# Define a custom Jinja2 filter to check if an image file exists
def image_exists(filename):
    path = os.path.join(app.static_folder, 'upload', filename)
    return os.path.exists(path)
    
# Add the custom filter to the Jinja2 environment
app.jinja_env.filters['image_exists'] = image_exists

# Create table
class doctors(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(500), nullable=False)
    email = db.Column(db.String(500), nullable=False)
    phone = db.Column(db.String(500), nullable=False)
    specialization = db.Column(db.String(500), nullable=False)
    password = db.Column(db.String(500), nullable=False)
    image = db.Column(db.String(500), nullable=True)
    
    # create a string
    def __repr__(self):
        return '<Name %r>' % self.name

# Create table
class patients(db.Model):
    pid = db.Column(db.Integer, primary_key=True)
    pname = db.Column(db.String(500), nullable=False)
    pemail = db.Column(db.String(500), nullable=False)
    pphone = db.Column(db.String(500), nullable=False)
    pstatus = db.Column(db.String(500), nullable=False)
    id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    sex = db.Column(db.String(500), nullable=False)
    pmarital = db.Column(db.String(500), nullable=True)
    pmedicine = db.Column(db.String(500), nullable=True)
    ifmedicine = db.Column(db.String(500), nullable=True)
    pbirth = db.Column(db.Date, nullable=False)
    report = db.Column(db.String(500), nullable=True)
    # create a string
    def __repr__(self):
        return '<Name %r>' % self.pname

class loginForm(FlaskForm):
    email = StringField("Email:", validators=[DataRequired()])
    password = StringField("Password:", validators=[DataRequired()])
    submit = SubmitField("Login")

class registerForm(FlaskForm):
    name = StringField("Name:", validators=[DataRequired()])
    email = StringField("Email:", validators=[DataRequired()])
    phone = StringField("Phone:", validators=[DataRequired()])
    password = StringField("Password:", validators=[DataRequired()])
    specialization = StringField("Specialization:", validators=[DataRequired()])
    image = FileField("Upload profile Photo:")
    submit = SubmitField("signup")

class patientForm(FlaskForm):
    name = StringField("Name:", validators=[DataRequired()])
    email = StringField("Email:", validators=[DataRequired()])
    phone = StringField("Phone:", validators=[DataRequired()])
    status = SelectField("Status:", choices=[('Normal', 'Normal'), ('Abnormal', 'Abnormal')] ,validators=[DataRequired()])
    birthdate = DateField("Birth Date:", format='%Y-%m-%d', validators=[DataRequired()])
    sex = SelectField("Sex:", choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    marital_status = SelectField("Marital Status:", choices=[('Single', 'Single'), ('Married', 'Married'), ('Divorced', 'Divorced')])
    medicine = SelectField("Medicine:", choices=[('Yes', 'Yes'), ('No', 'No')])
    if_medicine = TextAreaField("If yes, please list it:")
    report = TextAreaField("Report:", validators=[DataRequired()])
    submit = SubmitField("Add")

# Load your trained model
json_file = open("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Classification\\model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Classification\\model_weights.h5")
print("Loaded model from disk")
adam = Adam(learning_rate=0.0001)


def mcc(y_true, y_pred):
    y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = tf.keras.backend.sum(y_pos * y_pred_pos)
    tn = tf.keras.backend.sum(y_neg * y_pred_neg)

    fp = tf.keras.backend.sum(y_neg * y_pred_pos)
    fn = tf.keras.backend.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = tf.keras.backend.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + tf.keras.backend.epsilon())


model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy", t.Precision(), t.Recall(), mcc])
print('Model loaded. Check http://127.0.0.1:5000/')

def contour(image, mask):
    mask = mask[0][:, :, 0]
    # extract contour of white foreground in mask
    contours = measure.find_contours(mask, 0.5)
    # draw outline on image
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for contour in contours:
        coords = np.array(contour, dtype=np.int32)
        coords[:, [0, 1]] = coords[:, [1, 0]]
        draw.line(tuple(map(tuple, coords.tolist())), fill='yellow', width=3)
    img = img.resize((400, 400))
    return img

json_file_seg = open("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Segmentation\\Models\\Res-unet++\\final\\res-unet++_layers.json", 'r')
loaded_model_json_seg = json_file_seg.read()
json_file_seg.close()
model_seg = model_from_json(loaded_model_json_seg)
print("Loaded model from disk")
model_seg.load_weights("D:\\College\\Level 4\\First Term\\Graduation Project\\Models\\Segmentation\\Models\\Res-unet++\\final\\resunet++_weights.h5")
smooth = 1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-4)
model_seg.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', sm.metrics.iou_score, dice_coef])

def model_predict(img_path, model, model_seg):
    img = cv2.imread(img_path)

    # Preprocessing the image
    img = cv2.resize(img, (256, 256))
    img = cv2.medianBlur(img, 3)
    arr = []
    arr.append(img)
    arr = np.array(arr)

    preds = model.predict(arr)
    idx_cat = np.argmax(preds, axis=1)[0]
    tumor_type = ''
    if idx_cat == 1:
        tumor_type = 'polyps'
    else:
        tumor_type = 'normal'
    
    if tumor_type == 'polyps':
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = cv2.medianBlur(img, 3)
        predMask = model_seg.predict(np.expand_dims(img, axis=0), verbose=0)
        predMask = (predMask > 0.5).astype(np.bool_)
        annotated_image = contour(img, predMask)
        return tumor_type, annotated_image
    else:
        return tumor_type 

@login_manager.user_loader
def load_user(user_id):
    return doctors.query.get(int(user_id))

# Define a custom before_request function
@app.before_request
def before_request():
    if current_user.is_authenticated:
        # User is logged in, allow them to access the requested page
        pass
    else:
        # User is not logged in, redirect them to the login page
        if request.endpoint not in ['index', 'signup']:
            return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    # Login page
    form = loginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        user = doctors.query.filter_by(email=email, password=hashed_password).first()
        if user and user.password == hashed_password:
            login_user(user)
            session['id'] = user.id
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    session.clear()
    logout_user()
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    # signup page
    form = registerForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        phone = form.phone.data
        password = form.password.data
        specialization = form.specialization.data
        hashed_password  = hashlib.sha256(password.encode('utf-8')).hexdigest()
        new_doctor = doctors(name=name, email=email, phone=phone, password=hashed_password, specialization=specialization)
        db.session.add(new_doctor)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('signup.html', form=form)


@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    # home page
    doctor_id = session.get('id')
    patient_list = patients.query.filter_by(id=doctor_id).all()
    return render_template('index.html', patients=patient_list)


@app.route('/addpatient', methods=['GET', 'POST'])
def addpatient():
    # signup page
    form = patientForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        phone = form.phone.data
        status = form.status.data
        birth_date = form.birthdate.data
        sex = form.sex.data
        marital_status = form.marital_status.data
        medicine = form.medicine.data
        if_medicine = form.if_medicine.data
        report = form.report.data
        doctor_id = session.get('id')
        new_patient = patients(
            pname=name,
            pemail=email,
            pphone=phone,
            pstatus=status,
            id=doctor_id,
            pbirth=birth_date,
            sex=sex,
            pmedicine=medicine,
            ifmedicine=if_medicine,
            report=report,
            pmarital=marital_status
        )
        db.session.add(new_patient)
        db.session.commit()
        return redirect(url_for('home'))
    return render_template('patients.html', form=form)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    form = registerForm()
    doctor_id = session.get('id')
    # get the doctor object from the database
    doctor = doctors.query.get(doctor_id)

    # handle form submission
    if request.method == 'POST':
        # update the doctor information
        doctor.name = request.form['name']
        doctor.email = request.form['email']
        doctor.phone = request.form['phone']
        photo = request.files['image']
        if photo:
            basepath = os.path.dirname(__file__)
            photo_path = os.path.join(basepath, 'static\\upload', photo.filename)
            photo.save(photo_path)
            doctor.image = photo.filename
        doctor.specialization = request.form['specialization']
        db.session.commit()
        return redirect(url_for('profile', doctor_id=doctor.id))

    # render the template with the doctor object
    return render_template('profile.html', doctor=doctor, form=form)

@app.route('/update/<int:patient_id>', methods=['GET', 'POST'])
def update(patient_id):
    p = patients.query.get(patient_id)
    form = patientForm(obj=p)
    form.status.data = p.pstatus
    form.sex.data = p.sex
    form.marital_status.data = p.pmarital
    form.if_medicine.data = p.ifmedicine
    form.medicine.data = p.pmedicine
    form.report.data = p.report
    # handle form submission
    if request.method == 'POST':
        # update the doctor information
        p.pname = request.form['name']
        p.pemail = request.form['email']
        p.pphone = request.form['phone']
        p.sex = request.form['sex']
        p.pbirth = request.form['birthdate']
        p.pmarital = request.form['marital_status']
        p.pmedicine = request.form['medicine']
        p.pstatus = request.form['status']
        p.ifmedicine = request.form['if_medicine']
        p.report = request.form['report']
        db.session.commit()
        return redirect(url_for('home'))
    return render_template('update.html', patient=p, form=form, patient_id=patient_id)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model, model_seg)
        if isinstance(result, tuple):
            tumor_type, annotated_image = result
            buffered = BytesIO()
            annotated_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return jsonify({'result': tumor_type, 'image': img_str})
        else:
            return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
