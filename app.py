import os
from flask import Flask, render_template, url_for, redirect, flash, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import pandas as pd
from inference import ModelPredict
import google.generativeai as genai
from dotenv import load_dotenv
from utils import analyze_sentiment
app = Flask(__name__)

bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
app.config['SECRET_KEY'] = 'thisisasecretkey'
db = SQLAlchemy(app)

load_dotenv()

gemini_api_key = os.getenv("gemini_key")
genai.configure(api_key=gemini_api_key)
genai_model = genai.GenerativeModel('gemini-1.5-flash')

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

modelPredict = ModelPredict(
    model_path='lstm_model_weights.pth',
    scaler_path='scaler.pkl',
    data_path='stock_data.csv'
)
PROMPT = (
    "You are an expert trader/investor. You will be given a few statistics about a stock, "
    "and you will have to predict whether the stock will go up or down, and provide an explanation for your prediction.\n\n"
    "Statistics given will be:\n"
    "'open', 'volume', 'adjclose', 'RSIadjclose15', 'RSIvolume15', 'RSIadjclose25', 'RSIvolume25',\n"
    "'MACDadjclose15', 'MACDvolume15', 'MACDadjclose25', 'MACDvolume25',\n"
    "'emaadjclose5', 'emavolume5', 'emaadjclose10', 'emavolume10',\n"
    "'emaadjclose15', 'emavolume15', 'emaadjclose50', 'emavolume50'\n\n"
    "Here's the past 30 days of data for the stock:\n : {}\n\n"
    "And here's our model prediction:\n : {}\n\n"
    "Please provide a detailed explanation of the stock's performance based on the statistics provided also give the risk factor for what you predict. "
)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('display_stocks', ticker='ASML', start='2022-02-08', end='2022-03-22'))

            else:
                flash('Incorrect password. Please try again.', 'error')  
        else:
            flash('User not found. Please register first.', 'error')     
    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/recommend')
def recommend():
    return render_template('recommend.html')


@app.route('/analyze', methods=['POST'])
# @login_required
def analyze():
    data = request.json
    text = data.get('text', '')
    score, sentiment, recommendation = analyze_sentiment(text)
    return jsonify({
        'score': score,
        'sentiment': sentiment,
        'recommendation': recommendation
    })
        
        
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Successfully registered! Please log in.', 'success')  
        return redirect(url_for('login')) 

    return render_template('register.html', form=form)

df = pd.read_csv("stock_data.csv", parse_dates=["date"])
#df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date']) 

@app.route('/api/filter_stocks', methods=['GET'])
def filter_stocks():
    ticker = request.args.get('ticker')
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    if not ticker or not start_date or not end_date:
        return jsonify({'error': 'Missing required parameters'}), 400

    filtered = df[
        (df['ticker'] == ticker) &
        (df['date'] >= pd.to_datetime(start_date)) &
        (df['date'] <= pd.to_datetime(end_date))
    ]

    return filtered.to_json(orient="records") 

@app.route('/stocks')
@login_required
def display_stocks():
    ticker = request.args.get('ticker')
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    if not ticker or not start_date or not end_date:
        return "Please provide ticker, start, and end in the URL", 400

    filtered = df[
        (df['ticker'] == ticker) &
        (df['date'] >= pd.to_datetime(start_date)) &
        (df['date'] <= pd.to_datetime(end_date))
    ]

    tickers = df['ticker'].unique().tolist()

    columns_to_display = [
        'date', 'open', 'adjclose', 'volume', 'ticker',
        'RSIadjclose15', 'RSIvolume15'
    ]

    prediction, confidence = modelPredict.predict(start_date, ticker)

    if prediction[0] == 1:
        prediction_text = "Our model predicts that the stock will go up."
    else:
        prediction_text = "Our model predicts that the stock will go down."

    prompt = PROMPT.format(filtered.to_dict(orient='records'), prediction_text)
    response = genai_model.generate_content(prompt)
    print(response)

    return render_template('stocks.html', rows=filtered[columns_to_display].to_dict(orient='records'), current_ticker=ticker, start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        prediction_text=prediction_text, confidence=confidence[0],
        gemini_response = response.text
    )

if __name__ == "__main__":
    with app.app_context(): 
        db.create_all()
    app.run(debug=True)