from flask import Flask,render_template,request
import pickle as pk
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = pk.load(open('xgb.pkl', 'rb'))
scaler = pk.load(open('sc.pkl', 'rb'))  # Ensure you have a scaler.pkl saved
df_train = pk.load(open('df_model.pkl', 'rb'))  # Ensure you have df_train.pkl saved


# Prediction function
def predict_price(newob, model, scaler, df_train):
    input_values = newob[0]
     
    # Assigning input values
    brand_name = str(input_values[0])  
    vehicle_age = int(input_values[1])  
    km_driven = int(input_values[2])  
    seller_type = str(input_values[3])  
    fuel_type = str(input_values[4])  
    transmission_type = str(input_values[5])  
    mileage = float(input_values[6])  
    engine = int(input_values[7])  
    max_power = float(input_values[8])  
    seats = int(input_values[9])  

    input_dict = {
        "brand": brand_name,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df, columns=["brand", "seller_type", "fuel_type", "transmission_type"], dtype=int)

    # Handle missing columns
    missing_cols = set(df_train.drop(columns=[ "log_price"]).columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Align with training data
    input_df = input_df[df_train.drop(columns=["log_price"]).columns]
    input_data = scaler.transform(input_df)

    # Predict price
    predicted_price = model.predict(input_data)
    predicted_price = np.exp(predicted_price)  # Convert log price back to actual price

    return predicted_price[0]

app = Flask(__name__)

@app.route("/") #static page
def index():
    return render_template('index.html')

@app.route("/getprediction",methods=["POST"])
def getpredict():
    bn=request.form['bn']
    va=request.form['va']
    kd=request.form['kd']
    st=request.form['st'] 
    ft=request.form['ft'] 
    tt=request.form['tt'] 
    m=request.form['m'] 
    e=request.form['e']
    mp=request.form['mp'] 
    s=request.form['s']

    newob = [[bn, int(va), int(kd), st, ft, tt, float(m), int(e), float(mp), int(s)]]
    
    yp = predict_price(newob, model, scaler, df_train)
    return render_template('predict.html',data=yp)

if __name__ == "__main__":
    app.run(debug=True)