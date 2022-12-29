from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, TextAreaField, PasswordField
from wtforms.validators import DataRequired, Length, EqualTo

class UserCreateForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired('사용자 이름을 입력 해 주세요'), Length(min=3, max=25)])
    password1 = PasswordField('비밀번호', validators=[
        DataRequired('비밀번호를 입력 해 주세요'), EqualTo('password2', '비밀번호가 일치하지 않습니다')])
    password2 = PasswordField('비밀번호확인', validators=[DataRequired()])
    user_img = FileField('사용자 사진', validators=[FileRequired('사진을 업로드 해 주세요'), FileAllowed(['jpg', 'png'], 'Image Only!')])

class UserChangeForm(FlaskForm):
    password1 = PasswordField('비밀번호', validators=[EqualTo('password2', '비밀번호가 일치하지 않습니다')])
    password2 = PasswordField('비밀번호확인', validators=[])
    user_img = FileField('사용자 사진', validators=[FileAllowed(['jpg', 'png'], 'Image Only!')])


class UserLoginForm(FlaskForm):
    username = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    password = PasswordField('비밀번호', validators=[DataRequired()])


class ProductForm(FlaskForm):
    ID = StringField('상품번호')
    description = StringField('상품설명')
    img_path = StringField('상품이미지경로')