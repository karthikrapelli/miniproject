from flask import Flask, request, render_template
import numpy as np
import model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    
    # Create the input array
    input_data = np.array([[height, weight]])
    
    # Get the prediction
    prediction = model.predict(input_data)
    
    bmi_classes = {
        3: 'Obese Class 1',
        2: 'Overweight',
        0: 'Underweight',
        4: 'Obese Class 2',
        5: 'Obese Class 3',
        1: 'Normal Weight'
    }
    
    predicted_class = bmi_classes[prediction[0]]
    
    return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
