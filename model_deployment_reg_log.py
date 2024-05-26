from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

app = Flask(__name__)
api = Api(app, version='1.0', title='Movie Genre Classification', description='Movie Genre Classification')

modelo_reg_log = joblib.load(os.path.dirname(__file__) + '/modelo_regresion_logistica.pkl') 
vect = joblib.load(os.path.dirname(__file__) + '/vectorizador_tfidf.pkl')

# Definir nombres de columnas para las predicciones
cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama',
         'p_Family', 'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery',
          'p_News', 'p_Romance', 'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

parser = api.parser()

parser.add_argument('Plot', type=str, required=True, help='Write the plot of the movie', location='args')

resource_fields = api.model('Resource', {
       'Movie Genre': fields.String(description='Movie Genre')
})

@api.route('/predict')
class PrediccionGenre(Resource):
    @api.expect(parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()

        # Verificar variables inputs
        plot_input = args['Plot']

        # Aplicar el vectorizador TF-IDF al texto del argumento Plot
        plot_vectorizado = vect.transform([plot_input])
        
        # Realizar la predicción con el modelo Regresión Logística
        prediction_proba  = modelo_reg_log.predict_proba(plot_vectorizado)

        # Tomar la primera fila de prediction_proba (la única estimación)
        prob_row = prediction_proba[0]

        # Inicializar la lista de predicciones
        prediction = []

        # Iterar sobre las probabilidades predichas y sus índices
        for i, prob in enumerate(prob_row):
            # Verificar si la probabilidad para esta clase supera el umbral
            if prob > 0.5:
                # Agregar el nombre de la clase correspondiente a la lista de predicciones
                prediction.append(cols[i])
        
        # Devolver los géneros de las películas
        return {'Movie Genre': prediction}, 200
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)