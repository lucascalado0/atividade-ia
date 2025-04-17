from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cria o app Flask
app = Flask(__name__)

# Carrega o modelo
modelo = joblib.load('modelo_svm.pkl')

# Rota padrão
@app.route('/')
def home():
    return 'API do Modelo de Câncer de Mama - Online!'

# Rota para fazer previsão
@app.route('/predict', methods=['POST'])
def predict():
    dados = request.get_json()
    if not dados:
        return jsonify({'erro': 'Nenhum dado recebido'}), 400

    try:
        # Espera uma lista de 30 features como input
        entrada = np.array(dados['features']).reshape(1, -1)
        predicao = modelo.predict(entrada)
        resultado = 'Maligno' if predicao[0] == 1 else 'Benigno'
        return jsonify({'resultado': resultado})
    except Exception as e:
        return jsonify({'erro': str(e)}), 500

# Executa o app
if __name__ == '__main__':
    app.run(debug=True)