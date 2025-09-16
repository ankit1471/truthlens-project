# app.py - FINAL CORRECTED VERSION

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# --- 1. INITIAL APP AND DB SETUP ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-that-should-be-changed'

# THIS IS THE PART WE ARE FIXING
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'instance', 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- 2. LOAD ML MODEL AND PREPROCESSING FUNCTION ---
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    exit()

ps = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

# --- 3. DATABASE USER MODEL ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 4. AUTHENTICATION ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your username and password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'warning')
            return redirect(url_for('register'))
        new_user = User(
            username=username,
            password=generate_password_hash(password, method='pbkdf2:sha256')
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- 5. CORE APPLICATION ROUTES ---
@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        preprocessed_news = preprocess_text(news_text)
        vectorized_news = vectorizer.transform([preprocessed_news])
        prediction = model.predict(vectorized_news)
        result = "REAL" if prediction[0] == 1 else "FAKE"
        return render_template('index.html', prediction_result=result, news_text=news_text)
    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

# --- 6. CREATE DATABASE AND RUN APP ---
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)