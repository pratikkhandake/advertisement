

from flask import Flask, request, render_template, jsonify
from utils import get_predicted_sales
from config import port_number

app = Flask(__name__)

# Homepage API   
# for the sake of it

@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input values from the HTML form
    tv = float(request.form["tv"])
    radio = float(request.form["radio"])
    newspaper = float(request.form["newspaper"])

    # Get predicted sales using utils function
    predicted_sales = get_predicted_sales(tv, radio, newspaper)

    # Return the predicted sales
    return jsonify({"sales_prediction": predicted_sales})

if __name__ == "__main__":
    app.run(port=port_number,debug=False)
