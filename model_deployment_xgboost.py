from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd
import os
from xgboost import XGBRegressor

app = Flask(__name__)
api = Api(app, version='1.0', title='Used Vehicle Price Prediction', description='Used Vehicle Price Prediction')

#modelo_xgboost = joblib.load(os.path.dirname(__file__) + '/modelo_XGBoost.pkl') 

parser = api.parser()
parser.add_argument('Year', type=int, required=True, help='Year of the vehicle', location='args')
parser.add_argument('Mileage', type=int, required=True, help='Mileage of the vehicle', location='args')
parser.add_argument('Make', type=str, required=True, help='Make of the vehicle', location='args')
parser.add_argument('Model', type=str, required=True, help='Model of the vehicle', location='args')

resource_fields = api.model('Resource', {
       'Predicted Price': fields.Float(description='Predicted price of the vehicle')
})

@api.route('/predict')
class PrediccionPrecio(Resource):
    @api.expect(parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()
       
        data = {
            'Year': args['Year'],
            'Mileage': args['Mileage'],
            'Make': args['Make'],
            'Model': args['Model']
        }
        df = pd.DataFrame(data, index=[0])
        df['Make'] = df['Make'].astype('category')
        df['Model'] = df['Model'].astype('category')
        #prediction = modelo_xgboost.predict(df)[0]
        #data['Predicted Price'] = prediction
        data['Predicted Price'] = 989
        return data, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)