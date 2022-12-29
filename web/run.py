from pickle import TRUE
from re import A
from flask import Flask, render_template, request, session, g, redirect, template_rendered, url_for
from functools import wraps
from PIL import Image
from datetime import timedelta
import json
import random
import time
import threading
import requests

#from platformdirs import user_config_dir

import db
from form import UserCreateForm, UserLoginForm, ProductForm, UserChangeForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "12345678"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)

DB = db.DB()

allclothes = DB.GetAllCLothes()
global user_img_ori
global receive_ch
global fit_ch

# return list[dict{}]

@app.before_request
def load_logged_in_user():
    global user_img_ori
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = user_id
        path = 'static/images/user-image/'+user_id+'.jpg'
        user_img_ori = Image.open(path)

def pre_load():
    requests.get('http://203.250.148.132:5000/')

# 52.79.134.43:5000
@app.route('/')
def main():
    global receive_ch
    global fit_ch
    receive_ch = False
    fit_ch = False
    return redirect(url_for('main_home'))

@app.route('/product_fit')
def product_fit():
    global receive_ch
    global fit_ch
    global user_img_ori
    cloth = []
    cloth.append(request.args.get('item'))
    if fit_ch == False:
        fit_ch = True
        t = threading.Thread(target=fitting, args=(user_img_ori, cloth))
        t.start()
    if receive_ch == False:
        return render_template("loading.html")
    user_img = Image.open(user_img_ori.filename.replace('.jpg', 'Fitting.jpg'))
    fit_ch = False
    receive_ch = False
    return render_template("showimg.html", user_img=user_img.filename)

@app.route('/receive', methods=('GET', 'POST'))
def receive():
    global receive_ch
    if request.method == 'POST':
        user_img = request.files['image'] # 보낸 파일을 받아옴
        path = 'static/images/user-image/' + user_img.filename.replace('.jpg', 'Fitting.jpg')
        user_img.save(path)
        receive_ch = True

    return "success"

def fitting(user_img, checked):
    li = {'top':0, 'outwear':0}
    path = None
    for i in li:
        if i in checked[0]:
            path = i
    
    hf = open(user_img.filename)
    cf = open("static/images/item-image/" + path + "/" + checked[0] + ".jpg")

    upload = {'human_img': open(hf.name, 'rb'), 'cloth_img' : open(cf.name, 'rb')}
    res = requests.post('http://203.250.148.132:5000/', files=upload)

    return

def select_check(checked):
    sum = 0
    li = {'top':0, 'outwear':0}
    for i in li:
        for ch in checked:
            if i in ch:
                li[i]+=1
        sum += li[i]
    if sum == 0:
        return "옷을 선택해 주세요."
    if sum > 1:
        return "옷을 한 가지만 선택해 주세요"
    return None

@app.route('/main', methods=('GET', 'POST'))
def main_home():
    cloth = []
    value = request.args.get('cate')
    if request.method == 'GET' and  value != None:
        for type in allclothes :
            if value in type['ID'] :
                cloth.append(type)
    else :
        for type in allclothes :
                if "top" in type['ID'] :
                    cloth.append(type)

    return render_template("main.html", cloth = cloth)
@app.route('/product', methods=('GET', 'POST'))
def product():
    pform = ProductForm()
    item = request.args.get('item')
    pform = DB.GetCloth(item)
    if request.args.get('id') != None:
        id = request.args.get('id')
        DB.InsertUserCart(id, item)

    return render_template("product.html", pform = pform)
@app.route('/cart', methods=('GET', 'POST'))
def cart():
    cloth = []
    checked = []
    error = None
    global user_img_ori
    global receive_ch
    global fit_ch
    user_id = session.get('user_id')
    user_img = user_img_ori
    if session.get('user_id') == user_id:
        if request.form.get('Fitting') != None:
            for ch in request.form:
                if ch != 'Fitting':
                    checked.append(ch)
            error=select_check(checked)
            if error == None:
                t = threading.Thread(target=fitting, args=(user_img, checked))
                t.start()
                fit_ch = True
                return render_template("loading.html")

        if request.form.get('Remove') != None:
            for cl in request.form:
                rm_id = cl
                DB.DeleteUserCart(user_id, rm_id)
        
        if fit_ch == True and receive_ch == False:
            return render_template("loading.html")

        if receive_ch == True:
            user_img = Image.open(user_img.filename.replace('.jpg', 'Fitting.jpg'))
            receive_ch = False
            fit_ch = False

        cart = DB.GetUserCart(user_id)

        if cart != None:
            for c in cart:
                cloth.append(DB.GetCloth(c))
            return render_template("cart.html", cloth = cloth, error = error, user_img=user_img.filename)
    return redirect(url_for('main'))
@app.route('/signup/', methods=('GET', 'POST'))
def signup():
    form = UserCreateForm()
    error = None
    for field, errors in form.password1.errors:
            error = errors
    if request.method == 'POST' :
        if form.validate_on_submit():
            user = DB.GetUser(form.username.data)
            if user is None:
                img = form.user_img.data
                path = 'static/images/user-image/'+form.username.data+'.jpg'
                img.save(path)
                DB.InsertUser(form.username.data, form.password1.data, path)
                error = "가입이 성공되었습니다."
            else:
                error = "이미 존재하는 사용자 입니다."
        else:
            if form.username.data == "" :
                error = "사용자 이름을 입력해 주세요."
            elif form.password1.data == "" :
                error = "비밀번호를 입력해 주세요."
            elif form.password2.data == "" :
                error = "비밀번호를 다시 한번 입력해 주세요."
            elif form.user_img.data == None:
                error = "사용자 신체 이미지를 등록해 주세요."
            elif form.user_img.name[:-4] != ".jpg" and form.user_img.name[:-4] != ".png":
                error = "허용되지 않은 파일 확장자입니다. jpg 혹은 png로 등록해 주세요."

    return render_template('signup.html', form=form, error=error)

@app.route('/regist/', methods=('GET', 'POST'))
def regist():
    global user_img_ori
    form = UserChangeForm()
    if session.get('user_id') != None:
        if request.method == 'POST' and form.validate_on_submit():
            if form.user_img.data != None:
                img = form.user_img.data
                path = 'static/images/user-image/'+session['user_id']+'.jpg'
                img.save(path)
            if form.password1.data != '':
                DB.UpdateUser(session['user_id'], form.password1.data)
            return redirect(url_for('main'))
        return render_template("regist.html", form=form, user_img=user_img_ori.filename)
    return redirect(url_for('main'))

@app.route('/login/', methods=('GET', 'POST'))
def login():
    form = UserLoginForm()
    error = None
    url="/login/"
    if request.method == 'POST':
        if form.validate_on_submit():
            user = DB.GetUser(form.username.data)
            if not user:
                error = "존재하지 않는 사용자입니다."
            elif user['password'] != form.password.data:
                error = "비밀번호가 올바르지 않습니다."
            if error is None:
                session.clear()
                session['user_id'] = user['ID']
                return redirect(url_for('main_home'))
        else:
            if form.username.data == "" :
                error = "사용자 이름을 입력해 주세요."
            elif form.password.data == "" :
                error = "비밀번호를 입력해 주세요."
    
    return render_template('login.html', form=form, error=error, url=url)

@app.route('/logout')
def logout():
    session.clear()
    global user_img_ori
    user_img_ori.close()
    return redirect(url_for('main_home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)