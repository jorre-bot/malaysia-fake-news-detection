from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from predict_function import predict_news
from flask_cors import CORS
import time
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_talisman import Talisman
import secrets
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
import re
from datetime import timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['SESSION_COOKIE_SECURE'] = False  # Allow non-HTTPS in development
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Session expires after 30 minutes
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_ENABLED'] = True
app.config['WTF_CSRF_CHECK_DEFAULT'] = False  # Disable CSRF check for all routes by default
app.config['WTF_CSRF_SSL_STRICT'] = False  # Allow CSRF token without HTTPS in development

# Initialize extensions
db = SQLAlchemy(app)
csrf = CSRFProtect()
csrf.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Sila log masuk untuk mengakses halaman ini.'
login_manager.login_message_category = 'info'

# Security headers
Talisman(app,
    content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
        'style-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
        'img-src': ["'self'", "data:", "https:"],
        'font-src': ["'self'", "https:", "data:"],
    },
    force_https=True,
    strict_transport_security=True,
    session_cookie_secure=True,
    feature_policy={
        'geolocation': "'none'",
        'midi': "'none'",
        'notifications': "'none'",
        'push': "'none'",
        'sync-xhr': "'none'",
        'microphone': "'none'",
        'camera': "'none'",
        'magnetometer': "'none'",
        'gyroscope': "'none'",
        'speaker': "'none'",
        'vibrate': "'none'",
        'fullscreen': "'none'",
        'payment': "'none'",
    }
)

# Enable CORS with specific origins
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000", "http://192.168.100.20:5000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "X-CSRF-Token"],
        "supports_credentials": True,
        "expose_headers": ["X-CSRF-Token"]
    }
})

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    last_login = db.Column(db.DateTime)
    # Add relationship to detections
    detections = db.relationship('Detection', backref='user', lazy=True, order_by='Detection.created_at.desc()')

# Detection History Model
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)  # 'Real' or 'Fake'
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def validate_password(password):
    if len(password) < 8:
        return False, "Kata laluan mesti sekurang-kurangnya 8 aksara"
    if not re.search(r"[A-Za-z]", password):
        return False, "Kata laluan mesti mengandungi sekurang-kurangnya satu huruf"
    if not re.search(r"\d", password):
        return False, "Kata laluan mesti mengandungi sekurang-kurangnya satu nombor"
    return True, ""

def validate_input(text):
    """Validate the input text"""
    if not text or not text.strip():
        return False, "Text cannot be empty"
    return True, ""

@app.before_request
def before_request():
    if not request.is_secure and app.debug is False:
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)
    if current_user.is_authenticated:
        session.permanent = True
        app.permanent_session_lifetime = timedelta(minutes=30)

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            user.last_login = db.func.now()
            db.session.commit()
            
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('home'))
            
        flash('Nama pengguna atau kata laluan tidak sah', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("3 per minute")
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Kata laluan tidak sepadan', 'danger')
            return render_template('register.html')
            
        valid_pass, pass_msg = validate_password(password)
        if not valid_pass:
            flash(pass_msg, 'danger')
            return render_template('register.html')
            
        if User.query.filter_by(username=username).first():
            flash('Nama pengguna telah digunakan', 'danger')
            return render_template('register.html')
            
        if User.query.filter_by(email=email).first():
            flash('Emel telah digunakan', 'danger')
            return render_template('register.html')
            
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Pendaftaran berjaya! Sila log masuk.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Anda telah log keluar', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
@limiter.limit("10 per minute")
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })

@app.route('/history')
@login_required
def history():
    """Show user's detection history"""
    page = request.args.get('page', 1, type=int)
    per_page = 10
    detections = Detection.query.filter_by(user_id=current_user.id)\
        .order_by(Detection.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    return render_template('history.html', detections=detections)

@csrf.exempt
@app.route('/predict', methods=['POST'])
@limiter.limit("20 per minute")
@login_required
def predict():
    """
    Endpoint to predict if news is real or fake
    Expects JSON input with 'text' field
    """
    try:
        # Get the request data
        data = request.get_json()
        
        # Check if text is provided
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400
            
        # Get the text from request
        news_text = data['text']
        
        # Validate input
        valid, message = validate_input(news_text)
        if not valid:
            return jsonify({
                'error': message,
                'status': 'error'
            }), 400
            
        # Make prediction
        result = predict_news(news_text)
        
        # Save to history
        detection = Detection(
            user_id=current_user.id,
            text=news_text,
            prediction=result['prediction'],
            confidence=result['confidence']
        )
        db.session.add(detection)
        db.session.commit()
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'processed_text': result['processed_text'][:200] + '...' if len(result['processed_text']) > 200 else result['processed_text']
        }
        
        return jsonify(response)
        
    except Exception as e:
        # Log the error (you should set up proper logging)
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/batch-predict', methods=['POST'])
@limiter.limit("10 per minute")
@login_required
def batch_predict():
    """
    Endpoint to predict multiple news articles at once
    Expects JSON input with 'texts' field containing array of texts
    """
    try:
        # Get the request data
        data = request.get_json()
        
        # Check if texts array is provided
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'No texts provided',
                'status': 'error'
            }), 400
            
        # Get the texts from request
        news_texts = data['texts']
        
        # Check if input is list and validate size
        if not isinstance(news_texts, list) or len(news_texts) > 50:
            return jsonify({
                'error': 'Invalid input format or too many texts',
                'status': 'error'
            }), 400
            
        # Validate each text
        for text in news_texts:
            valid, message = validate_input(text)
            if not valid:
                return jsonify({
                    'error': message,
                    'status': 'error'
                }), 400
        
        # Make predictions for all texts
        results = []
        for text in news_texts:
            result = predict_news(text)
            results.append({
                'text': text[:200] + '...' if len(text) > 200 else text,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
        
        # Prepare response
        response = {
            'status': 'success',
            'results': results
        }
        
        return jsonify(response)
        
    except Exception as e:
        # Log the error (you should set up proper logging)
        print(f"Error in batch prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found', 'status': 'error'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 