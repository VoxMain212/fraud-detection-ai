from flask import Flask, request, render_template_string
import joblib
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

# 🔹 Загрузка модели и артефактов
MODEL_PATH = os.path.join("notebooks", "models", "fraud_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Файл {MODEL_PATH} не найден.\n"
        "Сначала запустите все ячейки в notebook/fraud_detection.ipynb, чтобы обучить и сохранить модель."
    )

artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
le_loc = artifacts['le_loc']
le_dev = artifacts['le_dev']
metrics = artifacts['metrics']

# 🔹 HTML-шаблон веб-интерфейса
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Fraud Detection Service</title>
    <style>
        :root { --primary: #2563eb; --success: #16a34a; --danger: #dc2626; --bg: #f8fafc; }
        body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); margin: 0; padding: 40px 20px; color: #1e293b; }
        .container { max-width: 500px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        h1 { text-align: center; margin-bottom: 5px; font-size: 24px; }
        .subtitle { text-align: center; color: #64748b; margin-bottom: 25px; font-size: 14px; }
        .form-group { margin-bottom: 16px; }
        label { display: block; margin-bottom: 6px; font-weight: 500; font-size: 14px; }
        input, select { width: 100%; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 15px; box-sizing: border-box; transition: 0.2s; }
        input:focus, select:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,0.15); }
        button { width: 100%; padding: 12px; background: var(--primary); color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 500; cursor: pointer; transition: 0.2s; margin-top: 10px; }
        button:hover { background: #1d4ed8; }
        .result-box { margin-top: 20px; padding: 16px; border-radius: 8px; text-align: center; font-weight: 500; display: none; }
        .result-box.safe { background: #dcfce7; color: var(--success); border: 1px solid #bbf7d0; display: block; }
        .result-box.fraud { background: #fee2e2; color: var(--danger); border: 1px solid #fecaca; display: block; }
        .metrics { margin-top: 25px; padding-top: 15px; border-top: 1px solid #e2e8f0; text-align: center; font-size: 12px; color: #64748b; }
        .error { background: #fff1f2; color: #be123c; padding: 12px; border-radius: 8px; margin-top: 15px; text-align: center; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Детектор мошенничества</h1>
        <p class="subtitle">Заполните параметры транзакции для анализа</p>
        
        <form method="POST">
            <div class="form-group">
                <label>💰 Сумма транзакции (₽)</label>
                <input type="number" step="0.01" name="amount" required placeholder="Например: 3500.50">
            </div>
            <div class="form-group">
                <label>⏰ Время операции (0–23)</label>
                <input type="number" name="time" min="0" max="23" step="1" required placeholder="Например: 14">
            </div>
            <div class="form-group">
                <label>📍 Местоположение</label>
                <select name="location" required>
                    {% for loc in locations %}<option value="{{ loc }}">{{ loc }}</option>{% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label>📱 Устройство</label>
                <select name="device" required>
                    {% for dev in devices %}<option value="{{ dev }}">{{ dev }}</option>{% endfor %}
                </select>
            </div>
            <button type="submit">🚀 Проверить транзакцию</button>
        </form>

        {% if result %}
        <div class="result-box {{ 'fraud' if result.is_fraud else 'safe' }}">
            {% if result.is_fraud %}
                ⚠️ <b>ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА</b><br>
            {% else %}
                ✅ <b>ТРАНЗАКЦИЯ БЕЗОПАСНА</b><br>
            {% endif %}
            Вероятность: <b>{{ "%.1f"|format(result.prob*100) }}%</b>
        </div>
        {% endif %}

        {% if error %}
        <div class="error">❌ Ошибка: {{ error }}</div>
        {% endif %}

        <div class="metrics">
            Модель: {{ model_name }} | ROC-AUC: {{ "%.3f"|format(auc) }} | F1-Score: {{ "%.3f"|format(f1) }}
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            amount = float(request.form["amount"])
            time = float(request.form["time"])
            location = request.form["location"]
            device = request.form["device"]

            # Кодирование категориальных признаков (используем те же LabelEncoder)
            loc_enc = le_loc.transform([location])[0]
            dev_enc = le_dev.transform([device])[0]

            # Формируем DataFrame в точном соответствии с обучением
            input_df = pd.DataFrame(
                [[amount, time, loc_enc, dev_enc]], 
                columns=["amount", "time", "location_enc", "device_enc"]
            )

            # Масштабирование и предсказание
            input_scaled = scaler.transform(input_df)
            pred_label = model.predict(input_scaled)[0]
            prob_fraud = model.predict_proba(input_scaled)[0][1]

            result = {"prob": prob_fraud, "is_fraud": bool(pred_label == 1)}

        except ValueError as e:
            error = f"Некорректные данные. Проверьте поля ввода. ({str(e)})"
        except Exception as e:
            error = f"Ошибка обработки запроса: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        locations=list(le_loc.classes_),
        devices=list(le_dev.classes_),
        result=result,
        error=error,
        model_name="Ensemble Classifier",
        auc=metrics.get("ROC-AUC", 0),
        f1=metrics.get("F1-Score", 0)
    )

if __name__ == "__main__":
    print("🌐 Запуск веб-сервиса: http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)