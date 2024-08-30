from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_language():
    if request.method == 'POST':
        text = request.form['text']
        detected_language = predict_language(text)  # Call the prediction function
        return render_template('result.html', language=detected_language)
    return render_template('index.html')

def predict_language(text):
    model = joblib.load('language_detector_model.pkl')
    cv = joblib.load('count_vectorizer.pkl')
    user_data = cv.transform([text]).toarray()
    prediction = model.predict(user_data)
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)