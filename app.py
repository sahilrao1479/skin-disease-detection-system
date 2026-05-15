import sqlite3
import re
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
app.secret_key = "secret123"

DB = "users.db"

# =========================
# DB SETUP
# =========================
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model/best_model.h5")

classes = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions"
]

# =========================
# VALIDATION
# =========================
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_doctor_id(doc):
    return doc.startswith("DOC") and len(doc) >= 6

def is_valid_password(pw):
    return len(pw) >= 8 and re.search(r"[!@#$%^&*(),.?\":{}|<>]", pw)

# =========================
# ROUTES
# =========================
@app.route('/')
def login():
    return render_template("login.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route('/logout')
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# =========================
# REGISTER
# =========================
@app.route('/register', methods=['POST'])
def do_register():
    user = request.form['username']
    password = request.form['password']
    confirm = request.form['confirm']

    if not (is_valid_email(user) or is_valid_doctor_id(user)):
        flash("Enter valid Email or Doctor ID (DOCxxx)")
        return redirect(url_for("register"))

    if not is_valid_password(password):
        flash("Password must be ≥8 chars & include special character")
        return redirect(url_for("register"))

    if password != confirm:
        flash("Passwords do not match")
        return redirect(url_for("register"))

    hashed_pw = generate_password_hash(password)

    try:
        conn = sqlite3.connect(DB)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user, hashed_pw))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        flash("User already exists")
        return redirect(url_for("register"))

    flash("Registration successful. Please login.")
    return redirect(url_for("login"))

# =========================
# LOGIN
# =========================
@app.route('/login', methods=['POST'])
def do_login():
    user = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = ?", (user,))
    result = cur.fetchone()
    conn.close()

    if result and check_password_hash(result[0], password):
        session['user'] = user
        return redirect(url_for("dashboard"))

    flash("Invalid credentials")
    return redirect(url_for("login"))

# =========================
# PREDICTION
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"})

    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    top = preds.argsort()[-2:][::-1]

    results = [{
        "disease": classes[i],
        "confidence": round(float(preds[i] * 100), 2)
    } for i in top]

    note = "✔ Reliable" if preds[top[0]] > 0.6 else "⚠ Low confidence"

    return jsonify({"predictions": results, "note": note})

if __name__ == "__main__":
    app.run(debug=True)