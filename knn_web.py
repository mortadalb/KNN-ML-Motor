import numpy as np
from sklearn import preprocessing
import pickle
import json
import jwt
import datetime
from functools import wraps
from flask import Flask, jsonify, request, make_response
from flask_restful import Resource, Api
from flask_httpauth import HTTPBasicAuth
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = "XXXXXXXXX"
api = Api(app)
auth = HTTPBasicAuth()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get('token')  # http://127.0.0.1:5000/route?token=xxxxxxxxxx
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
        except:
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(*args, **kwargs)
    return decorated

@app.route('/login')
def login():
    auth = request.authorization
    if auth and auth.username == 'xxxx' and auth.password == 'xxxxxxxxxx':
        token = jwt.encode({'user': auth.username, 'exp': datetime.utcnow() + timedelta(minutes=30)},
                           app.config['SECRET_KEY'])
        return jsonify({'token': token.decode('UTF-8')})
    return make_response('Could not verify!', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})

class HelloWorld(Resource):
    #@auth.login_required
    def get(self):
        return {"about":"Machine Learning (KNN) Web Service!"}

class Test(Resource):
    #@auth.login_required
    @token_required
    def get(self):
        return {"test":"Machine Learning (KNN) Web Service!"}

class LR_Predict(Resource):
    #@auth.login_required
    @token_required
    def post(self):
        params = [0,0,0,0,0,0,0,0,0,0]
        #{"parameters": {
        #    "insured_age": 47,
        #    "insured_nationality": "Bahraini",
        #    "source": "Direct",
        #    "pol_type": "Motor Third Party Liability Takaful",
        #    "business": "Renewal",
        #    "type_of_body": "Saloon",
        #    "vehicle_make": "BMW",
        #    "vehicle_model": "440i",
        #    "vehicle_age:" "1",
        #    "class_of_use": "Private"}
        #}
        parameters = request.get_json()
        json_data = json.loads(json.dumps(parameters))
        params[0] = json_data["parameters"]["insured_age"]
        params[1] = json_data["parameters"]["insured_nationality"]
        params[2] = json_data["parameters"]["source"]
        params[3] = json_data["parameters"]["pol_type"]
        params[4] = json_data["parameters"]["business"]
        params[5] = json_data["parameters"]["type_of_body"]
        params[6] = json_data["parameters"]["vehicle_make"]
        params[7] = json_data["parameters"]["vehicle_model"]
        params[8] = json_data["parameters"]["vehicle_age"]
        params[9] = json_data["parameters"]["class_of_use"]

        en_age = preprocessing.LabelEncoder()
        en_age.classes_ = np.load('insured_age.npy')
        try:
            insured_age = en_age.transform([params[0]])
            print('insured_age: ', insured_age)
        except:
            return jsonify({"status": {'result': 'Age not found!'}})
        en_nation = preprocessing.LabelEncoder()
        en_nation.classes_ = np.load('insured_nationality.npy')
        try:
            insured_nationality = en_nation.transform([params[1]])
            print('insured_nationality: ', insured_nationality)
        except:
            return jsonify({"status": {'result': 'Nationality not found!'}})
        en_source = preprocessing.LabelEncoder()
        en_source.classes_ = np.load('source.npy')
        try:
            source = en_source.transform([params[2]])
            print('source: ', source)
        except:
            return jsonify({"status": {'result': 'Source not found!'}})
        en_policy = preprocessing.LabelEncoder()
        en_policy.classes_ = np.load('pol_type.npy')
        try:
            pol_type = en_policy.transform([params[3]])
            print('pol_type: ', pol_type)
        except:
            return jsonify({"status": {'result': 'Policy Type not found!'}})
        en_business = preprocessing.LabelEncoder()
        en_business.classes_ = np.load('business.npy')
        try:
            business = en_business.transform([params[4]])
            print('business: ', business)
        except:
            return jsonify({"status": {'result': 'Business not found!'}})
        en_body = preprocessing.LabelEncoder()
        en_body.classes_ = np.load('type_of_body.npy')
        try:
            type_of_body = en_body.transform([params[5]])
            print('type_of_body: ', type_of_body)
        except:
            return jsonify({"status": {'result': 'Body Type not found!'}})
        en_make = preprocessing.LabelEncoder()
        en_make.classes_ = np.load('vehicle_make.npy')
        try:
            vehicle_make = en_make.transform([params[6]])
            print('vehicle_make: ', vehicle_make)
        except:
            return jsonify({"status": {'result': 'Vehicle Make not found!'}})
        en_model = preprocessing.LabelEncoder()
        en_model.classes_ = np.load('vehicle_model.npy')
        try:
            vehicle_model = en_model.transform([params[7]])
            print('vehicle_model: ', vehicle_model)
        except:
            return jsonify({"status": {'result': 'Vehicle Model not found!'}})
        en_year = preprocessing.LabelEncoder()
        en_year.classes_ = np.load('vehicle_age.npy')
        try:
            vehicle_age = en_year.transform([params[8]])
            print('vehicle_age: ', vehicle_age)
        except:
            return jsonify({"status": {'result': 'Vehicle Age not found!'}})
        en_use = preprocessing.LabelEncoder()
        en_use.classes_ = np.load('class_of_use.npy')
        try:
            class_of_use = en_use.transform([params[9]])
            print('class_of_use: ', class_of_use)
        except:
            return jsonify({"status": {'result': 'Class of Use not found!'}})

        pickle_in = open("motor_knn_model.pickle", "rb")
        knn = pickle.load(pickle_in)
        x = np.array([[insured_age,
                       insured_nationality,
                       source,
                       pol_type,
                       business,
                       type_of_body,
                       vehicle_make,
                       vehicle_model,
                       vehicle_age,
                       class_of_use]])
        print('x = ', x)
        x = x.reshape(1, -1)
        predicted = knn.predict(x)
        print('predicted = ', predicted)

        en_lr_ratio = preprocessing.LabelEncoder()
        en_lr_ratio.classes_ = np.load('loss_ratio.npy')
        y = en_lr_ratio.inverse_transform(predicted)
        response = {'prediction': y[0]}
        print('y = ', y)

        return jsonify(response)

if __name__ == '__name__':
    app.run(debug=True)

api.add_resource(HelloWorld, '/')
api.add_resource(Test, '/test')
api.add_resource(LR_Predict, '/status')