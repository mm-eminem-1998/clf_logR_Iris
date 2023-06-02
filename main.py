
from flask import Flask, jsonify, render_template, request

from Project_App.utils import IrisDataset

# Creating instance here
app = Flask(__name__)

@app.route("/")
def hello_flask():
    print("Welcome to Flower Species Prediction System...")
    return render_template("index.html")

@app.route("/predict_Species", methods = ["POST","GET"])
def get_prediction():
    if request.method == "GET":
        print("We are in GET method")

        SepalLengthCm = float(request.args.get("SepalLengthCm"))
        SepalWidthCm = float(request.args.get("SepalWidthCm"))
        PetalLengthCm = float(request.args.get("PetalLengthCm"))
        PetalWidthCm = float(request.args.get("PetalWidthCm"))

        print("SepalLengthCm,  SepalWidthCm,  PetalLengthCm,  PetalWidthCm, smoker, region", SepalLengthCm,  SepalWidthCm,  PetalLengthCm, PetalWidthCm)

        iris = IrisDataset(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
        flower_species = iris.get_prediction()

        if flower_species == 0:
            text = "Iris-Setosa"
        elif flower_species == 1:
            text = "Iris-versicolor"
        else:
            text ="Iris-virginica"

    return render_template("index.html",prediction = text)

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port= 5005, debug = False)

