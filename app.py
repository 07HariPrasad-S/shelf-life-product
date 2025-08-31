from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
from datetime import datetime, timedelta
import re
import os

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS so frontend can call API

# Load dataset
df = pd.read_csv("large_product_dataset.csv")

# Clean dataset product names (remove units like (1kg), (200g), etc.)
df['Clean Name'] = df['Product Name'].apply(
    lambda x: re.sub(r"\(.*?\)", "", x).strip().lower()
)

# Load trained ML model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "✅ Product Expiry Prediction API is running! Visit /frontend for UI."

@app.route("/get_info", methods=["GET"])
def get_info():
    product = request.args.get("product")
    manufacture_date = request.args.get("manufacture_date")

    if not product:
        return jsonify({"error": "Please provide 'product'"}), 400

    # Normalize user input: remove units and lowercase
    clean_product = re.sub(r"\(.*?\)", "", product).strip().lower()

    # 🔹 Lookup in dataset
    product_row = df[df['Clean Name'] == clean_product]

    if not product_row.empty:
        shelf_life_days = int(product_row['Shelf Life (Days)'].values[0])
    else:
        # 🔹 Predict using ML model if not found
        try:
            shelf_life_days = int(model.predict([product])[0])
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    result = {
        "product": clean_product.capitalize(),
        "shelf_life_days": shelf_life_days
    }

    # 🔹 If manufacture date provided → calculate expiry
    if manufacture_date:
        try:
            mfg_date = datetime.strptime(manufacture_date, "%Y-%m-%d")
            expiry_date = mfg_date + timedelta(days=shelf_life_days)
            result["manufacture_date"] = manufacture_date
            result["expiry_date"] = expiry_date.strftime("%Y-%m-%d")
            result["status"] = "Expired" if datetime.now() > expiry_date else "Fresh"
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    return jsonify(result)

# ✅ Serve frontend (index.html) from same folder
@app.route("/frontend")
def frontend():
    return send_from_directory(os.getcwd(), "index.html")

if __name__ == "__main__":
    app.run(debug=True)
