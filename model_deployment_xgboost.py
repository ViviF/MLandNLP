from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd
import os
from xgboost import XGBRegressor

app = Flask(__name__)
api = Api(app, version='1.0', title='Used Vehicle Price Prediction', description='Used Vehicle Price Prediction')

# Extraer la ruta del archivo actual y Cargar el modelo XGBoost en esta ruta 
modelo_xgboost = joblib.load(os.path.dirname(__file__) + '/modelo_XGBoost_region.pkl') 

def clasificar_estado(estado):
    estado = estado.strip()
    if estado in ['AK', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'OR', 'UT', 'WA', 'WY']:
        return 'West'
    elif estado in ['CT', 'DE', 'MA', 'MD', 'ME', 'NH', 'NJ', 'NY', 'PA', 'RI']:
        return 'Northeast'
    elif estado in ['IA', 'IL', 'IN', 'KS', 'MI', 'MN', 'MO', 'ND', 'NE', 'OH', 'SD', 'WI']:
        return 'Midwest'
    elif estado in ['AL', 'AR', 'FL', 'GA', 'KY', 'LA', 'MS', 'NC', 'SC', 'TN', 'VA', 'WV']:
        return 'Southeast'
    elif estado in ['AZ', 'NM', 'OK', 'TX']:
        return 'Southwest'
    else:
        return 'Error'

# Definir parser para analizar los argumentos de la solicitud
resource_fields = api.model('Resource', {'result': fields.Float(description='Used Vehicle Price Prediction')})

# Definir la clase de recurso para manejar las solicitudes de predicción
@api.route('/predict')
class PrediccionPrecio(Resource):
    @api.expect(api.model('Price', {
        'Year': fields.Integer(required=True, description='Year of the vehicle'),
        'Mileage': fields.Integer(required=True, description='Mileage of the vehicle'),
        'State': fields.String(required=True, description='State of the vehicle'),
        'Make': fields.String(required=True, description='Make of the vehicle'),
        'Model': fields.String(required=True, description='Model of the vehicle'),
    }))
    @api.marshal_with(resource_fields)
    def post(self):
        # Obtener los datos de la solicitud
        #data = request.json
        
        # Obtener los datos de la solicitud del formulario
        year = int(request.form['Year'])
        mileage = int(request.form['Mileage'])
        state = request.form['State']
        make = request.form['Make']
        model = request.form['Model']

        # Obtener la región correspondiente
        region = clasificar_estado(state)
    
        # Crear un DataFrame con los datos de entrada
        df = pd.DataFrame({'Year': [year], 'Mileage': [mileage], 'Make': [make], 'Model': [model], 'Region': [region]})
    

        # Categorizar las variables
        #df['Make'] = df['Make'].astype('category')
        #df['Model'] = df['Model'].astype('category')
        #df['Region'] = df['Region'].astype('category')

        # Realizar la predicción con el modelo XGBoost
        prediction = modelo_xgboost.predict(df)[0]
        
        # Devolver el precio estimado
        return {'Predicted Price': prediction}, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
