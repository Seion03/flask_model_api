from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('profitability_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    worker_salary = data['worker_salary']
    raw_material_cost = data['raw_material_cost']
    targeted_materials_produced = data['targeted_materials_produced']
    
    # Prepare the input for the model
    input_features = [[worker_salary, raw_material_cost, targeted_materials_produced]]
    
    # Make prediction
    prediction = model.predict(input_features)

    return jsonify({'predicted_profit': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
