from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def welcome_page():
    return render_template('welcome.html')  # Render the welcome.html template

@app.route('/data')
def data_page():
    return render_template('data.html')  # Render the data.html template

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')  # Render the analytics.html template

@app.route('/predict', methods=['GET', 'POST'])
def prediction_page():
    if request.method == 'POST':
        # Handle form submission
        student_id = request.form.get('studentId')
        study_hours = request.form.get('studyHours')
        sleep_hours = request.form.get('sleepHours')
        social_hours = request.form.get('socialHours')
        stress_level = request.form.get('stressLevel')

        # Perform prediction logic here (placeholder)
        prediction_result = f"Prediction for Student {student_id}: Study Hours = {study_hours}, Sleep Hours = {sleep_hours}, Social Hours = {social_hours}, Stress Level = {stress_level}"

        return render_template('predict.html', prediction_result=prediction_result)
    return render_template('predict.html')  # Render the predict.html template

if __name__ == '__main__':
    app.run(debug=True)