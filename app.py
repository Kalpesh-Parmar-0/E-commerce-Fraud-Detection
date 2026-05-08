from flask import Flask, render_template, request, jsonify
from src.pipeline.predict_pipeline import PredictPipeline, CustomClass

app = Flask(__name__)

@app.route('/', methods= ['GET', 'POST'])
def prediction_data ():
    if request.method == 'GET':
        return render_template("home.html") 
    else:
        data = CustomClass(
            TransactionAmt=float(request.form.get('TransactionAmt')),
            ProductCD= request.form.get('ProductCD'), 
            card1= int(request.form.get('card1')), 
            card2= float(request.form.get('card2')), 
            card3= float(request.form.get('card3')), 
            card4= request.form.get('card4'), 
            card5= float(request.form.get('card5')), 
            card6= request.form.get('card6'), 
            addr1= float(request.form.get('addr1')), 
            addr2= float(request.form.get('addr2')), 
            P_emaildomain= request.form.get('P_emaildomain'), 
            R_emaildomain= request.form.get('R_emaildomain'), 
            DeviceType= request.form.get('DeviceType'), 
            DeviceInfo= request.form.get('DeviceInfo')
        )

        final_data = data.get_data_as_data_frame()

        pipeine_prediction = PredictPipeline()
        pred , proba= pipeine_prediction.predict(final_data)

        result = pred[0]
        confidence = proba[0]

        if result == 1:
            return render_template("results.html", final_result = "Fraud", probability=round(confidence, 4))
        elif result == 0:
            return render_template("results.html", final_result = "Not Fraud", probability=round(confidence, 4))

if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug = True)