from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_mail import Mail, Message
from flask_bcrypt import Bcrypt
import joblib
import pandas as pd
import google.generativeai as genai
import json
import os
import secrets
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from google.api_core import exceptions
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO

# Initialize Flask app
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or secrets.token_hex(16)
bcrypt = Bcrypt(app)

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')
mail = Mail(app)

# Configure Google Generative AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
MODEL_NAME = 'gemini-1.5-flash'
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    raise ValueError(f"Failed to initialize Gemini model: {str(e)}")

# Paths and upload config
USER_DB_PATH = 'users.json'
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# SQLite database setup
DB_PATH = 'predictions.db'

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                type TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

# Initialize database
init_db()

def serialize_data(obj):
    """Convert non-serializable objects to JSON-serializable formats."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()  # Convert Timestamp to ISO string
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_data(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_data(item) for item in obj]
    return obj

def save_prediction(username, prediction_type, data):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        serialized_data = serialize_data(data)  # Convert Timestamps to strings
        cursor.execute(
            'INSERT INTO predictions (username, type, data) VALUES (?, ?, ?)',
            (username, prediction_type, json.dumps(serialized_data))
        )
        conn.commit()
        return cursor.lastrowid

def get_latest_prediction(username, prediction_type):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT data FROM predictions WHERE username = ? AND type = ? ORDER BY created_at DESC LIMIT 1',
            (username, prediction_type)
        )
        result = cursor.fetchone()
        return json.loads(result[0]) if result else None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_users():
    if os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, 'r') as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(USER_DB_PATH, 'w') as file:
        json.dump(users, file)

def send_email(to_email, subject, body, html=None):
    try:
        msg = Message(subject, recipients=[to_email])
        msg.body = body
        if html:
            msg.html = html
        mail.send(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def load_model(model_path='xgboost_model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    artifacts = joblib.load(model_path)
    return artifacts['pipeline'], artifacts['label_encoder']

def predict_new_data(input_data, model_path='xgboost_model.pkl'):
    try:
        pipeline, label_encoder = load_model(model_path)
        new_data = pd.DataFrame([input_data])
        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        new_data['hour'] = new_data['timestamp'].dt.hour
        new_data['minute'] = new_data['timestamp'].dt.minute
        new_data['day_of_week'] = new_data['timestamp'].dt.dayofweek
        processed_data = new_data.drop(['timestamp', 'dest_ip'], axis=1)
        prediction = pipeline.predict(processed_data)
        return {
            'input_features': input_data,
            'predicted_attack': label_encoder.inverse_transform(prediction)[0]
        }
    except Exception as e:
        return {'error': f"Prediction failed: {str(e)}"}

def generate_prediction_description(prediction_result):
    attack_type = prediction_result.get('predicted_attack', 'Unknown')
    user_input = f"""The predicted attack type is '{attack_type}'. Generate a detailed report section for this attack type, including:
- **Characteristics**: What defines this attack?
- **Potential Impact**: What harm can it cause?
- **Mitigation Strategies**: How to prevent or respond?
Format as markdown with clear sections."""
    try:
        response = model.generate_content(
            user_input,
            generation_config={"temperature": 0.6, "max_output_tokens": 1000}
        )
        return response.text.strip() if response.text else "Error: Unable to generate description."
    except exceptions.InvalidArgument:
        return "Error: Invalid input or model configuration."
    except exceptions.QuotaExceeded:
        return "Error: API quota exceeded. Please try again later."
    except exceptions.GoogleAPICallError as e:
        return f"Error: API call failed - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def process_csv_file(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return {'error': 'CSV file is empty'}
        
        required_columns = ['timestamp', 'source_ip', 'dest_ip', 'source_port', 'dest_port', 
                           'protocol', 'packet_count', 'byte_count', 'duration']
        if not all(col in df.columns for col in required_columns):
            return {'error': 'CSV missing required columns: ' + ', '.join(required_columns)}
        
        results = []
        unique_attacks = set()
        errors = []
        for idx, row in df.iterrows():
            try:
                input_data = {
                    'timestamp': pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                    'source_ip': str(row['source_ip']),
                    'dest_ip': str(row['dest_ip']),
                    'source_port': int(row['source_port']),
                    'dest_port': int(row['dest_port']),
                    'protocol': str(row['protocol']),
                    'packet_count': int(row['packet_count']),
                    'byte_count': int(row['byte_count']),
                    'duration': float(row['duration'])
                }
                if input_data['packet_count'] < 0 or input_data['byte_count'] < 0 or input_data['duration'] < 0:
                    raise ValueError("Numeric fields cannot be negative")
                
                prediction_result = predict_new_data(input_data)
                if 'error' not in prediction_result:
                    unique_attacks.add(prediction_result['predicted_attack'])
                    results.append(prediction_result)
                else:
                    errors.append(f"Row {idx + 1}: {prediction_result['error']}")
            except (ValueError, KeyError) as e:
                errors.append(f"Row {idx + 1}: Invalid data - {str(e)}")
        
        if not results:
            return {'error': 'No valid predictions made. Errors: ' + '; '.join(errors)}
        
        descriptions = {attack: generate_prediction_description({'predicted_attack': attack}) 
                        for attack in unique_attacks}
        
        return {'results': results, 'descriptions': descriptions, 'errors': errors}
    except Exception as e:
        return {'error': f"Error processing CSV: {str(e)}"}
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        users = load_users()
        if username in users:
            flash('Username already exists', 'danger')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            users[username] = {'password': hashed_password, 'email': email}
            save_users(users)
            subject = "Registration Successful"
            body = f"Hello {username},\n\nYour registration was successful. You can now log in.\n\nThank you!"
            send_email(email, subject, body)
            flash('Registration successful. Check your email.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username not in users:
            flash('Invalid username or password', 'danger')
        else:
            stored_password = users[username]['password']
            try:
                if stored_password.startswith('$2b$'):
                    if bcrypt.check_password_hash(stored_password, password):
                        session['username'] = username
                        flash('Login successful', 'success')
                        return redirect(url_for('index'))
                    else:
                        flash('Invalid username or password', 'danger')
                else:
                    if stored_password == password:
                        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                        users[username]['password'] = hashed_password
                        save_users(users)
                        session['username'] = username
                        flash('Login successful. Your password has been updated for security.', 'success')
                        return redirect(url_for('index'))
                    else:
                        flash('Invalid username or password', 'danger')
            except ValueError as e:
                flash(f'Login failed: {str(e)}. Please re-register or reset your password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logout successful', 'success')
    return redirect(url_for('landing'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        return redirect(url_for('landing'))
    users = load_users()
    user_email = users.get(session['username'], {}).get('email', '')
    
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                result = process_csv_file(filepath)
                if 'error' in result:
                    flash(result['error'], 'danger')
                    return redirect(url_for('index'))
                
                # Save to database instead of session
                prediction_id = save_prediction(session['username'], 'batch', result)
                session['last_batch_prediction_id'] = prediction_id
                
                report_content = "Batch Network Attack Prediction Report\n====================\n"
                report_html = """
                <html>
                <body style='font-family: Arial, sans-serif;'>
                    <h1>Batch Network Attack Prediction Report</h1>
                    <h2>Predictions</h2>
                    <table style='border-collapse: collapse; width: 100%;'>
                        <tr style='background: #3b82f6; color: white;'>
                            <th style='padding: 10px; border: 1px solid #ccc;'>#</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Attack Type</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Timestamp</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Source IP</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Dest IP</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Source Port</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Dest Port</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Protocol</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Packets</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Bytes</th>
                            <th style='padding: 10px; border: 1px solid #ccc;'>Duration</th>
                        </tr>
                """
                for idx, pred in enumerate(result['results'], 1):
                    report_content += f"**Prediction {idx}**:\n"
                    report_content += f"- Attack Type: {pred['predicted_attack']}\n"
                    report_content += f"- Timestamp: {pred['input_features']['timestamp']}\n"
                    report_content += f"- Source IP: {pred['input_features']['source_ip']}\n"
                    report_content += f"- Destination IP: {pred['input_features']['dest_ip']}\n"
                    report_content += f"- Source Port: {pred['input_features']['source_port']}\n"
                    report_content += f"- Destination Port: {pred['input_features']['dest_port']}\n"
                    report_content += f"- Protocol: {pred['input_features']['protocol']}\n"
                    report_content += f"- Packet Count: {pred['input_features']['packet_count']}\n"
                    report_content += f"- Byte Count: {pred['input_features']['byte_count']}\n"
                    report_content += f"- Duration: {pred['input_features']['duration']} seconds\n\n"
                    report_html += f"""
                        <tr>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{idx}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['predicted_attack']}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['input_features']['timestamp']}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['input_features']['source_ip']}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['input_features']['dest_ip']}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['input_features']['source_port']}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['input_features']['dest_port']}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['input_features']['protocol']}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['input_features']['packet_count']}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['input_features']['byte_count']}</td>
                            <td style='padding: 10px; border: 1px solid #ccc;'>{pred['input_features']['duration']}</td>
                        </tr>
                    """
                report_html += "</table><h2>Analysis</h2>"
                report_content += "**Analysis**:\n"
                for attack, desc in result['descriptions'].items():
                    report_content += f"### {attack}\n{desc}\n\n"
                    report_html += f"<h3>{attack}</h3><pre>{desc}</pre>"
                report_content += "====================\nStay vigilant!"
                report_html += "</body></html>"
                
                if not send_email(user_email, "Batch Network Attack Prediction Report", report_content, html=report_html):
                    flash('Failed to send email notification', 'warning')
                
                if result['errors']:
                    flash('Some rows could not be processed: ' + '; '.join(result['errors']), 'warning')
                
                return redirect(url_for('batch_report'))
            else:
                flash('Invalid file type. Please upload a CSV file.', 'danger')
        else:
            try:
                sample_input = {
                    'timestamp': pd.to_datetime(request.form['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                    'source_ip': request.form['source_ip'],
                    'dest_ip': request.form['dest_ip'],
                    'source_port': int(request.form['source_port']),
                    'dest_port': int(request.form['dest_port']),
                    'protocol': request.form['protocol'],
                    'packet_count': int(request.form['packet_count']),
                    'byte_count': int(request.form['byte_count']),
                    'duration': float(request.form['duration'])
                }
                if sample_input['packet_count'] < 0 or sample_input['byte_count'] < 0 or sample_input['duration'] < 0:
                    raise ValueError("Numeric fields cannot be negative")
            except (ValueError, KeyError) as e:
                flash(f"Invalid input: {str(e)}", 'danger')
                return redirect(url_for('index'))
            
            prediction_result = predict_new_data(sample_input)
            if 'error' in prediction_result:
                flash(prediction_result['error'], 'danger')
                return redirect(url_for('index'))
            
            description = generate_prediction_description(prediction_result)
            prediction_data = {
                'input_features': prediction_result['input_features'],
                'predicted_attack': prediction_result['predicted_attack'],
                'description': description
            }
            # Save to database
            prediction_id = save_prediction(session['username'], 'single', prediction_data)
            session['last_prediction_id'] = prediction_id
            
            report_content = f"""Network Attack Prediction Report
====================
**Attack Type**: {prediction_result['predicted_attack']}
**Input Details**:
- Timestamp: {prediction_result['input_features']['timestamp']}
- Source IP: {prediction_result['input_features']['source_ip']}
- Destination IP: {prediction_result['input_features']['dest_ip']}
- Source Port: {prediction_result['input_features']['source_port']}
- Destination Port: {prediction_result['input_features']['dest_port']}
- Protocol: {prediction_result['input_features']['protocol']}
- Packet Count: {prediction_result['input_features']['packet_count']}
- Byte Count: {prediction_result['input_features']['byte_count']}
- Duration: {prediction_result['input_features']['duration']}
**Analysis**:
{description}
====================
Stay vigilant!"""
            if not send_email(user_email, "Network Attack Prediction Report", report_content):
                flash('Failed to send email notification', 'warning')
            
            return render_template('result.html',
                                   input_features=prediction_result['input_features'],
                                   predicted_attack=prediction_result['predicted_attack'],
                                   description=description)
    
    return render_template('index.html')

@app.route('/report')
def report():
    if 'username' not in session:
        return redirect(url_for('landing'))
    if 'last_prediction_id' not in session:
        flash('No single prediction available for report', 'warning')
        return redirect(url_for('index'))
    
    prediction = get_latest_prediction(session['username'], 'single')
    if not prediction:
        flash('No single prediction available for report', 'warning')
        return redirect(url_for('index'))
    
    return render_template('report.html',
                           input_features=prediction['input_features'],
                           predicted_attack=prediction['predicted_attack'],
                           description=prediction['description'])

@app.route('/batch_report')
def batch_report():
    if 'username' not in session:
        return redirect(url_for('landing'))
    if 'last_batch_prediction_id' not in session:
        flash('No batch prediction available for report', 'warning')
        return redirect(url_for('index'))
    
    batch_prediction = get_latest_prediction(session['username'], 'batch')
    if not batch_prediction:
        flash('No batch prediction available for report', 'warning')
        return redirect(url_for('index'))
    
    return render_template('batch_report.html',
                           results=batch_prediction['results'],
                           descriptions=batch_prediction['descriptions'],
                           errors=batch_prediction['errors'])

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('landing'))
    
    users = load_users()
    user_info = users.get(session['username'], {})
    
    # Retrieve recent predictions from database
    recent_single_prediction = get_latest_prediction(session['username'], 'single')
    recent_batch_prediction = get_latest_prediction(session['username'], 'batch')
    
    return render_template('dashboard.html',
                           username=session['username'],
                           email=user_info.get('email', ''),
                           recent_single_prediction=recent_single_prediction,
                           recent_batch_prediction=recent_batch_prediction)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/download_report/<report_type>')
def download_report(report_type):
    if 'username' not in session:
        return redirect(url_for('landing'))
    
    if report_type == 'single' and 'last_prediction_id' not in session:
        flash('No single prediction available to download', 'warning')
        return redirect(url_for('dashboard'))
    if report_type == 'batch' and 'last_batch_prediction_id' not in session:
        flash('No batch prediction available to download', 'warning')
        return redirect(url_for('dashboard'))
    
    prediction = get_latest_prediction(session['username'], report_type)
    if not prediction:
        flash(f'No {report_type} prediction available to download', 'warning')
        return redirect(url_for('dashboard'))
    
    output_format = request.args.get('format', 'txt')
    
    if report_type == 'single':
        report_content = f"""Network Attack Prediction Report
====================
**Attack Type**: {prediction['predicted_attack']}
**Input Details**:
- Timestamp: {prediction['input_features']['timestamp']}
- Source IP: {prediction['input_features']['source_ip']}
- Destination IP: {prediction['input_features']['dest_ip']}
- Source Port: {prediction['input_features']['source_port']}
- Destination Port: {prediction['input_features']['dest_port']}
- Protocol: {prediction['input_features']['protocol']}
- Packet Count: {prediction['input_features']['packet_count']}
- Byte Count: {prediction['input_features']['byte_count']}
- Duration: {prediction['input_features']['duration']}
**Analysis**:
{prediction['description']}
====================
Stay vigilant!"""
        
        if output_format == 'pdf':
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            story.append(Paragraph("Network Attack Prediction Report", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"<b>Attack Type:</b> {prediction['predicted_attack']}", styles['Normal']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Input Details:</b>", styles['Heading2']))
            
            data = [
                ["Timestamp", prediction['input_features']['timestamp']],
                ["Source IP", prediction['input_features']['source_ip']],
                ["Destination IP", prediction['input_features']['dest_ip']],
                ["Source Port", str(prediction['input_features']['source_port'])],
                ["Destination Port", str(prediction['input_features']['dest_port'])],
                ["Protocol", prediction['input_features']['protocol']],
                ["Packet Count", str(prediction['input_features']['packet_count'])],
                ["Byte Count", str(prediction['input_features']['byte_count'])],
                ["Duration", str(prediction['input_features']['duration'])]
            ]
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Analysis:</b>", styles['Heading2']))
            for line in prediction['description'].split('\n'):
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("Stay vigilant!", styles['Normal']))
            
            doc.build(story)
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name="single_prediction_report.pdf", mimetype='application/pdf')
        else:
            return send_file(
                BytesIO(report_content.encode('utf-8')),
                as_attachment=True,
                download_name="single_prediction_report.txt",
                mimetype='text/plain'
            )
    
    elif report_type == 'batch':
        batch_prediction = prediction
        report_content = "Batch Network Attack Prediction Report\n====================\n"
        
        for idx, pred in enumerate(batch_prediction['results'], 1):
            report_content += f"**Prediction {idx}**:\n"
            report_content += f"- Attack Type: {pred['predicted_attack']}\n"
            report_content += f"- Timestamp: {pred['input_features']['timestamp']}\n"
            report_content += f"- Source IP: {pred['input_features']['source_ip']}\n"
            report_content += f"- Destination IP: {pred['input_features']['dest_ip']}\n"
            report_content += f"- Source Port: {pred['input_features']['source_port']}\n"
            report_content += f"- Destination Port: {pred['input_features']['dest_port']}\n"
            report_content += f"- Protocol: {pred['input_features']['protocol']}\n"
            report_content += f"- Packet Count: {pred['input_features']['packet_count']}\n"
            report_content += f"- Byte Count: {pred['input_features']['byte_count']}\n"
            report_content += f"- Duration: {pred['input_features']['duration']} seconds\n\n"
        
        report_content += "**Analysis**:\n"
        for attack, desc in batch_prediction['descriptions'].items():
            report_content += f"### {attack}\n{desc}\n\n"
        report_content += "====================\nStay vigilant!"
        
        if output_format == 'pdf':
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            story.append(Paragraph("Batch Network Attack Prediction Report", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Predictions:</b>", styles['Heading2']))
            
            data = [["#", "Attack Type", "Timestamp", "Source IP", "Dest IP", "Source Port", 
                     "Dest Port", "Protocol", "Packets", "Bytes", "Duration"]]
            for idx, pred in enumerate(batch_prediction['results'], 1):
                data.append([
                    str(idx),
                    pred['predicted_attack'],
                    pred['input_features']['timestamp'],
                    pred['input_features']['source_ip'],
                    pred['input_features']['dest_ip'],
                    str(pred['input_features']['source_port']),
                    str(pred['input_features']['dest_port']),
                    pred['input_features']['protocol'],
                    str(pred['input_features']['packet_count']),
                    str(pred['input_features']['byte_count']),
                    str(pred['input_features']['duration'])
                ])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Analysis:</b>", styles['Heading2']))
            for attack, desc in batch_prediction['descriptions'].items():
                story.append(Paragraph(f"<b>{attack}</b>", styles['Heading3']))
                for line in desc.split('\n'):
                    story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 12))
            story.append(Paragraph("Stay vigilant!", styles['Normal']))
            
            doc.build(story)
            buffer.seek(0)
            return send_file(buffer, as_attachment=True, download_name="batch_prediction_report.pdf", mimetype='application/pdf')
        else:
            return send_file(
                BytesIO(report_content.encode('utf-8')),
                as_attachment=True,
                download_name="batch_prediction_report.txt",
                mimetype='text/plain'
            )

if __name__ == "__main__":
    app.run(debug=True)