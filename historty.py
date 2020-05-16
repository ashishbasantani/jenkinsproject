from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import sequence

app = Flask(__name__)

def depression(x):
    if(x>0.5):
        return 1
    else:
        return 0


@app.route('/prediction', methods=['GET','POST'])
def get_tasks():
	if request.method == "POST":
		gender = request.form['gender']
		age = request.form['age']
		maritalstatus = request.form['maritalstatus']
		noofchild = request.form['noofchild']
		edu = request.form['edu']
		totalmembers = request.form['totalmembers']
		savings = request.form['savings']
		alcohol = request.form['alcohol']
		tobacco = request.form['tobacco']
		medicalexp = request.form['medicalexp']
		educationexp = request.form['educationexp']
		socialexp = request.form['socialexp']
		otherexp = request.form['otherexp']
		income = request.form['income']
		proportionofsick = request.form['proportionofsick']
		childsick = request.form['childsick']
		investment = request.form['investment']

	else:
		gender = request.args.get('gender')
		age = request.args.get('age')
		maritalstatus = request.args.get('maritalstatus')
		noofchild = request.args.get('noofchild')
		edu = request.args.get('edu')
		totalmembers = request.args.get('totalmembers')
		savings = request.args.get('savings')
		alcohol = request.args.get('alcohol')
		tobacco = request.args.get('tobacco')
		medicalexp = request.args.get('medicalexp')
		educationexp = request.args.get('educationexp')
		socialexp = request.args.get('socialexp')
		otherexp = request.args.get('otherexp')
		income = request.args.get('income')
		proportionofsick = request.args.get('proportionofsick')
		childsick = request.args.get('childsick')
		investment = request.args.get('investment')


	model = joblib.load('PatientHistoryDepressionPrediction.pk1')
	input1 = [gender,age,maritalstatus,noofchild,edu,totalmembers,savings,alcohol,tobacco,medicalexp,educationexp,socialexp,otherexp,income,proportionofsick,childsick,investment]
	input1 = np.array(input1)
	input1 = input1.reshape(1,17)

	predict = model.predict(input1)

	ans = depression(predict)

	return jsonify([{"prediction":ans}])


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')


#http://localhost:5000/prediction?gender=1&age=28.0&maritalstatus=1&noofchild=1&edu=2&totalmembers=50&savings=30&alcohol=1&tobacco=0&medicalexp=2&educationexp=10&socialexp=3&otherexp=6&income=30&proportionofsick=3&childsick=3&investment=3