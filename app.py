# app.py - Updated Version with Login/Register

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os # Added for database path handling

# --- 1. INITIAL APP AND DB SETUP ---

# Initialize Flask app
app = Flask(__name__)

# Secret key is needed for session management and flash messages
# Use DATABASE_URL from environment variable if available, otherwise use local sqlite
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'instance', 'users.db') 
# Configure the database URI
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to login page if user is not logged in

# --- 2. LOAD ML MODEL AND PREPROCESSING FUNCTION ---

# Load the trained model and vectorizer
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    exit()

# Define the text preprocessing function
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

# User loader function for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 4. AUTHENTICATION ROUTES (LOGIN, REGISTER, LOGOUT) ---

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
            flash('Logged in successfully!', 'success')
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

        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'warning')
            return redirect(url_for('register'))
        
        # Create new user with hashed password
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
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# --- 5. CORE APPLICATION ROUTES (HOME, PREDICT, ABOUT) ---

@app.route('/')
@login_required # This decorator protects the page
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required # Also protect the prediction endpoint
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        preprocessed_news = preprocess_text(news_text)
        vectorized_news = vectorizer.transform([preprocessed_news])
        prediction = model.predict(vectorized_news)
        
        result = "REAL" if prediction[0] == 1 else "FAKE"
        
        return render_template('index.html', prediction_result=result, news_text=news_text)
    # Redirect to home if accessed via GET
    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

# --- 6. CREATE DATABASE AND RUN APP ---

if __name__ == '__main__':
    # Ensure the instance folder exists
    if not os.path.exists('instance'):
        os.makedirs('instance')
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', debug=True)