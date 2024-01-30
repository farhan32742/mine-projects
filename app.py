from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("random_forest.pkl", 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
       
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    input_features_scaled = scaler.transform(features)
    prediction = model.predict(input_features_scaled)
    print('predictio value is ', prediction)
 
    
    return render_template("index.html", prediction_text = "The Person salary is {}".format(prediction))

if __name__ == "__main__":
       app.run(debug=True)
    