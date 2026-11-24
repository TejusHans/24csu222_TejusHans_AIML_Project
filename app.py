from pathlib import Path
from datetime import datetime
import sqlite3
import torch
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    'Original BERT': ROOT / 'trained_model_original',
    'Improved BERT': ROOT / 'trained_model_experiment2',
}
for name, path in MODEL_PATHS.items():
    if not (path / 'config.json').exists():
        raise RuntimeError(f'{name} not found at {path}. Check your model backup folders.')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS = {}
for name, path in MODEL_PATHS.items():
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True).to(DEVICE)
    model.eval()
    MODELS[name] = (tokenizer, model)
DB = ROOT / 'predictions.db'


def connect():
    conn = sqlite3.connect(DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT, news TEXT, prediction TEXT, confidence REAL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS comparison_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL, news TEXT NOT NULL,
        original_prediction TEXT NOT NULL, original_confidence REAL NOT NULL,
        improved_prediction TEXT NOT NULL, improved_confidence REAL NOT NULL
    )''')
    return conn


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    idx = int(torch.argmax(probs).item())
    return {'label': str(model.config.id2label[idx]),
            'confidence': round(float(probs[idx].item()) * 100, 2)}


@app.route('/', methods=['GET', 'POST'])
def home():
    results = None
    error = None
    news = ''
    if request.method == 'POST':
        news = request.form.get('news', '').strip()
        if len(news) < 20:
            error = 'Please enter at least 20 characters of news text.'
        elif len(news) > 10000:
            error = 'Please limit the text to 10,000 characters.'
        else:
            results = {name: predict(news, *pair) for name, pair in MODELS.items()}
            original = results['Original BERT']
            improved = results['Improved BERT']
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with connect() as conn:
                conn.execute('''INSERT INTO comparison_history
                    (created_at, news, original_prediction, original_confidence,
                     improved_prediction, improved_confidence)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (now, news, original['label'], original['confidence'],
                     improved['label'], improved['confidence']))
    with connect() as conn:
        history = conn.execute('''SELECT created_at, news, original_prediction,
            original_confidence, improved_prediction, improved_confidence
            FROM comparison_history ORDER BY id DESC LIMIT 10''').fetchall()
        total = conn.execute('SELECT COUNT(*) FROM comparison_history').fetchone()[0]
    return render_template('index.html', results=results, error=error,
                           news=news, history=history, total=total)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
