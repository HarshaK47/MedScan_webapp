from flask import Flask, render_template, request

import tensorflow
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import os
import sys
import cv2
from PIL import Image, ImageOps
import time
from werkzeug.utils import secure_filename





from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length, ValidationError
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user





import skimage

import glob
import cv2
import math
import imageio

import statistics
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage import data,io,filters
from skimage.filters import threshold_mean
from skimage.morphology import binary_dilation
from skimage import feature
from skimage.morphology import skeletonize_3d
from collections import Iterable



import warnings
warnings.filterwarnings("ignore")









app = Flask(__name__)





app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'



class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('Remember me')
    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if not existing_user_username:
            raise ValidationError(
                "Invalid Username"
            )

class RegisterForm(FlaskForm):
    email = StringField('Email ID', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                "This Username already exist. "
            )






@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/postapp', methods=['GET'])
def postapp():
    return redirect("http://localhost:3000", code=302)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return redirect(url_for('login'))
        #return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'

    return render_template('signup.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index_temp.html', name=current_user.username)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))








# @app.route("/")
# def hello_world():
#     return render_template('index_temp.html')

@app.route("/about", methods=['GET'])
def about():
    return render_template('index1.html')

@app.route("/semiauto",methods=['GET'])
def semiauto():

    return render_template('semiAuto.html');


@app.route("/processSemi",methods=['POST'])
def processSemi():

    f = request.files['file']

    # function to display the coordinates of
    # of the points clicked on the image
    xparams=[]
    yparams=[]

    def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
            print(x, ' ', y)
            xparams.append(x)
            yparams.append(y)
 
        # displaying the coordinates
        # on the image window
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, str(x) + ',' +
                        str(y), (x,y), font,
                        1, (255, 255, 255), 2)
            cv2.imshow('image', img)
 
    # checking for right mouse clicks    
        if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
            print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = img[y, x, 0]
            g = img[y, x, 1]
            r = img[y, x, 2]
            cv2.putText(img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x,y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image', img)
    
    basewala = "E:\\ALL_minor_copy\\images\\"
    actualimg = f.filename
    
    img = cv2.imread(basewala+actualimg)
 
    # displaying the image
    cv2.imshow('image', img)
 
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
 
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
 
    # close the window
    cv2.destroyAllWindows()

    pointp1x=xparams[0]
    pointp2x=xparams[1]
    pointvx=xparams[2]
    pointl1x=xparams[3]
    pointl2x=xparams[4]

    pointp1y=yparams[0]
    pointp2y=yparams[1]
    pointvy=yparams[2]
    pointl1y=yparams[3]
    pointl2y=yparams[4]

    a1 = abs(pointp1y-pointvy)
    b1=abs(pointvx-pointp1x)
    c1=abs(pointp2y-pointvy)
    d1=abs(pointp2x-pointvx)

    theta1 = a1/b1
    theta2 = c1/d1

    angle1 = math.degrees(math.atan(theta1))
    angle2 = math.degrees(math.atan(theta2))

    sulcusAngle = 180-angle1-angle2

    distance_1 = math.sqrt( ((pointp1x-pointvx)**2)+((pointp1y-pointvy)**2) )
    distance_2 = math.sqrt( ((pointp2x-pointvx)**2)+((pointp2y-pointvy)**2) )
    facet_asymmetry_1 = distance_1/distance_2
    facet_asymmetry_2 = distance_2/distance_1
    
    if(facet_asymmetry_1 < facet_asymmetry_2):
        fcfinal = facet_asymmetry_1
    else:
        fcfinal = facet_asymmetry_2


    if(distance_1 < distance_2):
        vector1 = ((-pointp2y+pointvy))/((pointp2x-pointvx))
        vector2 = ((pointl1y-pointl2y))/((pointl2x-pointl1x))
        ltiangle = math.degrees(math.atan((abs(vector1-vector2))/(abs(1 + vector1*vector2))))
        
    else:
        vector1 = ((-pointp1y+pointvy))/((-pointp1x+pointvx))
        vector2 = ((pointl1y-pointl2y))/((-pointl1x+pointl2x))
        ltiangle = math.degrees(math.atan((abs(vector1-vector2))/(abs(1 + vector1*vector2))))



    return render_template('semiAutooutput.html',p1x=pointp1x,p2x=pointp2x,vx=pointvx,l1x=pointl1x,l2x=pointl2x,p1y=pointp1y,p2y=pointp2y,vy=pointvy,l1y=pointl1y,l2y=pointl2y, sulcus=sulcusAngle, facet=fcfinal,ltiAngle = ltiangle)





@app.route("/report", methods=['POST'])
def printReport():
    pName=request.form['patientName']
    pId=request.form['patientid']
    # imgurl = request.form['imgurl']
    imgurl2 = request.form['imgurl2']
    sulcus = request.form['sulcusAngle']
    facetAs = request.form['facetA']
    ltiANGLE = request.form['Ltiangle']
    return render_template('report.html', pname=pName,pid=pId,imgUrl2=imgurl2,sulcusang=sulcus,facetasy=facetAs,ltiAng=ltiANGLE)



@app.route("/process" ,methods=['POST'])
def printImage():

    json_file = open("ResUNetpp.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

#Loading the model_weights
    model.load_weights("ResUNetpp_19.08.h5")
    f = request.files['file']

        # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    print(f.filename)
    f.save(file_path)

#Reading the image from a specified path
    img = cv2.imread(file_path)
#Conversion to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Resize the image to (320,320)
    img = cv2.resize(img, (320,320))

#Save the image as an array
    imgs_np = np.asarray(img)

#Normalization 
    X = np.asarray(imgs_np, dtype=np.float32)/255

    X = X.reshape(1, X.shape[0], X.shape[1], -1)
# print(X.shape)

#Prediction made after using the model_weights
    y_pred = model.predict(X)
# print(y_pred.shape)
    

    plt.subplot(121)
    plt.imshow(X[0,:,:,:])
    # plt.savefig('static/images/new_plot1.png')

    


    plt.subplot(122)
    plt.imshow(y_pred[0,:,:,0], cmap='gray',interpolation='none')
    tosavefig1 = 'static/images/'
    ct = datetime.datetime.now()
    ct=str(ct)
    xt=ct.replace(" ","_")
    yt=xt.replace(":","_")
    zt=yt.replace(".","_")
    ft=zt.replace("-","_")
    tosavefig1+=ft
    tosavefig1+='.png'
    
    plt.savefig(tosavefig1)
    # imageio.imwrite(tosavefig1,y_pred);

    plt.clf()
    plt.cla()
    plt.close()
    

    y_pred = np.reshape(y_pred,(X.shape[1],X.shape[2]))
    y_pred*=255
    y_pred = y_pred.astype(np.uint8)

    origin = [0,0]
    refvec = [0,1]

    #Function to flatten a list
    def flatten(lis):
         for item in lis:
             if isinstance(item, Iterable) and not isinstance(item, str):
                 for x in flatten(item):
                     yield x
             else:        
                 yield item

#Angle Computation
    def get_angle(p0, p1, p2):
    
        if p2 is None:
            p2 = p1 + np.array([1, 0])
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)

        cosine_angle = np.dot(v0, v1) / (np.linalg.norm(v0)* np.linalg.norm(v1))

        angle = np.arccos(cosine_angle)
        return abs(np.degrees(angle))

#Sorting the coordinate points of the boundary in clockwise direction with just ONE argument
    def clockwiseangle_and_distance(point):
    # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
    # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector

    arr = []

    thresh = threshold_mean(y_pred)
# Taking a matrix of size 5 as the kernel
    kernel = np.ones((3,3), np.uint8)

    img_erosion = cv2.erode(y_pred, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    binary = img_dilation > thresh
    
    edges = filters.sobel(binary)
    edge_map = edges[0:int(y_pred.shape[1]/2.2), 0:y_pred.shape[0]]
    edge_map_lower = edges[int(y_pred.shape[1]/2):y_pred.shape[1], 0:y_pred.shape[0]]
    
#Upper Edge Map
    edge_map = np.asarray(edge_map)
    pixels = np.argwhere(edge_map > 0)
    
    sorted(pixels, key=clockwiseangle_and_distance)
#print(pixels)

#Finding M and M'
    max_y = max(pixels[:,0])
#print(max_y)
    list_max_y = []

    for i in range(len(pixels)):
        if pixels[i,0] == max_y:
            list_max_y.append(pixels[i,:])

    list_max_y = np.asarray(list_max_y)
#print(list_max_y)
    min_x = min(list_max_y[:,1])
#print(min_x)
    max_x = max(list_max_y[:,1])
#print(max_x)
    mean_x = int((min_x+max_x)/2)
    # print(max_y,min_x)
    # print(max_y,max_x)

    
    #Peak (P1, P2) and Valley (V) Points in Upper Half Edge Map
    peak_1 = []
    peak_2 = []
    min_list1 = []
    count_1 = 0
    min_list2 = []
    count_2 = 0
    for i in range(len(pixels)):
        if min_x<=pixels[i,1]<mean_x:
            peak_1.append(pixels[i,:])
        if mean_x<= pixels[i,1]<=max_x:
            peak_2.append(pixels[i,:])

#Finding First Peak Point
    peak_1 = np.asarray(peak_1)
#print("List of first half points: "+str(peak_1))
    min_1 = np.min(peak_1[:,0])
#print("List of points with minimum y value: "+str(min_1))
    for i in peak_1:
        if i[0] == min_1:
            count_1 = count_1+1
            min_list1.append(i)
    min_list1 = np.asarray(min_list1)
#print("List of highest points in first half: "+str(min_list1))
    if len(min_list1)>1:
    #median = int(statistics.median(min_list1[:,1])) 
        min_value = np.min(min_list1[:,1])
        P1 = [min_1,min_value]
        P1 = np.asarray(P1)
        min_1y = P1[1]
    else:
        P1 = min_list1
        P1 = np.asarray(P1)
        min_1y = P1[0,1]
    P1 = list(flatten(P1))
    # print("Peak 1 for Patient: "+ str(P1))

#Finding Second Peak Point
    peak_2 = np.asarray(peak_2)
#print("List of second half points: "+str(peak_2))
    min_2 = np.min(peak_2[:,0])
#print("List of points with minimum y value: "+str(min_2))
    for i in peak_2:
        if i[0] == min_2:
            count_2 = count_2+1
            min_list2.append(i)
    min_list2 = np.asarray(min_list2)
#print("List of highest points in second half: "+str(min_list2))
    if len(min_list2)>1:
    #median_1 = int(statistics.median(min_list2[:,1])) 
        max_value = np.max(min_list2[:,1])
        P2 = [min_2,max_value]
        P2 = np.asarray(P2)
        min_2y = P2[1]
    else:
        P2 = min_list2
        P2 = np.asarray(P2)
        min_2y = P2[0,1]
    P2 = list(flatten(P2))
    # print("Peak 2 for Patient: " + str(P2))

    #Finding Valley Point
    in_between = []
    for i in pixels:
            j = i[1]
            if (min_1y < j < min_2y):
                in_between.append(i)

    low = []
    in_between = np.asarray(in_between)
    in_max = np.max(in_between[:,0])

    valley_list = []
    count_3 = 0
    for i in in_between:
        if i[0] == in_max:
            count_3 = count_3+1
            valley_list.append(i)
    valley_list = np.asarray(valley_list)

    if len(valley_list)>1:
        median_2 = int(statistics.median(valley_list[:,1])) 
        V = [in_max,median_2]
        V = np.asarray(V)
    else:
        V = valley_list
        V = np.asarray(V)
    V = list(flatten(V))
    # print("Valley Point for Patient: " + str(V))
    angle = get_angle(P1,V,P2)
    distance_1 = math.sqrt( ((P1[0]-V[0])**2)+((P1[1]-V[1])**2) )
    distance_2 = math.sqrt( ((P2[0]-V[0])**2)+((P2[1]-V[1])**2) )
    facet_asymmetry_1 = distance_1/distance_2
    facet_asymmetry_2 = distance_2/distance_1
    # print("\033[1mSulcus Angle for Patient: \033[0m" + str(angle))

    if facet_asymmetry_1 < facet_asymmetry_2:
        # print("\033[1mFacet Asymmetry: \033[0m" + str(facet_asymmetry_1))
        fa_acem = str(facet_asymmetry_1)
    else:
        # print("\033[1mFacet Asymmetry: \033[0m" + str(facet_asymmetry_2))
        fa_acem = str(facet_asymmetry_2)

    thresh_1 = threshold_mean(y_pred)
    binary_1 = y_pred > thresh_1
    edges_1 = filters.sobel(binary_1)
    edges_1 = np.asarray(edges_1)
    pixels_1 = np.argwhere(edges_1 > 0)
    
    sorted(pixels_1, key=clockwiseangle_and_distance)

    lower_1 = []
    lower_2 = []
    max_list1 = []
    counter_1 = 0
    max_list2 = []
    counter_2 = 0
    for i in range(len(pixels_1)):
        if min_x<=pixels_1[i,1]<mean_x:
            lower_1.append(pixels_1[i,:])
        if mean_x<= pixels_1[i,1]<=max_x:
            lower_2.append(pixels_1[i,:])
    
    #Finding Lowest Points in lower half
    lower_1 = np.asarray(lower_1)
#print("List of first half lower points: "+str(lower_1))
    max_1 = np.max(lower_1[:,0])
#print("List of points with maximum y value: "+str(max_1))
    for i in lower_1:
        if i[0] == max_1:
            counter_1 = counter_1+1
            max_list1.append(i)
    max_list1 = np.asarray(max_list1)
#print("List of lowest points in first half: "+str(max_list1))
    if len(max_list1)>1:
        max_value = np.max(max_list1[:,1])
        L1 = [max_1,max_value]
        L1 = np.asarray(L1)
        max_1y = L1[1]
    else:
        L1 = max_list1
        L1 = np.asarray(L1)
        max_1y = L1[0,1]
    L1 = list(flatten(L1))
    # print("Lower Point 1 for Patient: " + str(L1))

    #Finding Lowest Points in lower half
    lower_2 = np.asarray(lower_2)
#print("List of first half lower points: "+str(lower_2))
    
    max_2 = np.max(lower_2[:,0])
#print("List of points with maximum y value: "+str(max_2))

    for i in lower_2:
        if i[0] == max_2:
            counter_2 = counter_1+1
            max_list2.append(i)
    max_list2 = np.asarray(max_list2)
#print("List of lowest points in second half: "+str(max_list2))

    if len(max_list2)>1:
    #median_1 = int(statistics.median(min_list2[:,1])) 
        max_value1 = np.max(max_list2[:,1])
        L2 = [max_2,max_value1]
        L2 = np.asarray(L2)
        max_2y = L2[1]
    else:
        L2 = max_list2
        L2 = np.asarray(L2)
        max_2y = L2[0,1]

    L2 = list(flatten(L2))
    # print("Lower Point 2 for Patient: " + str(L2))
    
#print("Distance 1: "+str(distance_1))
#print("Distance 2: "+str(distance_2))

    if distance_1 < distance_2:
        v1 = np.array(V) - np.array(P2)
        v2 = np.array(L1) - np.array(L2)
    #print(v1,v2)
    #print(np.linalg.norm(v1))
    #print(np.linalg.norm(v2))
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1)* np.linalg.norm(v2))

        LTI = np.arccos(cosine_angle)
        LTI = abs(np.degrees(LTI))
    #print("Distance 2", LTI)
    
    else:
        v1 = np.array(P1) - np.array(V)
        v2 = np.array(L1) - np.array(L2)
    #print(v1,v2)
    #print(np.linalg.norm(v1))
    #print(np.linalg.norm(v2))
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1)* np.linalg.norm(v2))
    #print(cosine_angle)
        LTI = np.arccos(cosine_angle)
    #print(LTI)
        LTI = abs(np.degrees(LTI))
    #print("Distance 1",LTI)

    # print("\033[1mLateral Trochlear Inclination: \033[0m"+str(LTI))
    
    arr1 = []
    arr1.append(L1)
    arr1.append(L2)
    arr1.append(P1)
    arr1.append(P2)
    arr1.append(V)

    for item in arr1:
        cv2.drawMarker(edge_map, (item[0], item[1]),(0,0,255), markerType=cv2.MARKER_STAR, 
        markerSize=40, thickness=3, line_type=cv2.LINE_AA)
    edges*=255
    edges = edges.astype(np.uint8) 
    arr1 = np.array(arr1)
#print(arr1)
    plt.figure(figsize=(6,6))
    plt.imshow(edges)
    plt.scatter(arr1[:,1], arr1[:,0], color = 'red')
    print(arr1)
    plt.annotate("P1", (arr1[2,1] , arr1[2,0]), color='white')
    plt.annotate("P2", (arr1[3,1], arr1[3,0]), color='white')
    plt.annotate("V", (arr1[4,1], arr1[4,0]), color='white')
    plt.annotate("L1", (arr1[0,1], arr1[0,0]), color='white')
    plt.annotate("L2", (arr1[1,1], arr1[1,0]), color='white')
    tosavefig2 = 'static/images/'
    ct = datetime.datetime.now()
    ct=str(ct)
    xt=ct.replace(" ","_")
    yt=xt.replace(":","_")
    zt=yt.replace(".","_")
    ft=zt.replace("-","_")
    tosavefig2+=ft
    tosavefig2+='.png'

    plt.savefig(tosavefig2)

    plt.clf()
    plt.cla()
    plt.close()
    
    imgurl1 = '/'
    imgurl1+=tosavefig1
    imgurl2='/'
    imgurl2+=tosavefig2

    










    return render_template('output.html', name = 'new_plot',url2=imgurl2, url1 =imgurl1, maxY=max_y, maxX=max_x, minX=min_x,peak1=str(P1),peak2=str(P2),valley=str(V), sulcusangle=str(angle),facet=fa_acem,lower1=str(L1), lower2 = str(L2), lateral=str(LTI))



@app.route("/calculateparameters" ,methods=['POST'])
def parameters():
    print("Hello")


    

if __name__ == "__main__":
    app.run(debug=True, port=8000)


     
