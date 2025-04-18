from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Cria o app Flask
app = Flask(__name__)

# Carrega o modelo
modelo = joblib.load(r'C:\Users\Lucas Vinicius\Desktop\IA\modelo_mlp.pkl')


# Rota padrão
@app.route('/')
def home():
    return 'API do Modelo de Câncer - Online!'

from flask import Flask, request, render_template
import numpy as np
import pickle  # ou joblib, dependendo do que você usou para salvar o modelo

app = Flask(__name__)

@app.route('/teste', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Captura os dados do formulário
            radius_mean = float(request.form.get("radius"))
            texture_mean = float(request.form.get("texture"))
            perimeter_mean = float(request.form.get("perimeter"))
            area_mean = float(request.form.get("area"))
            smoothness_mean = float(request.form.get("smoothness"))
            compactness_mean = float(request.form.get("compactness"))
            concavity_mean = float(request.form.get("concavity"))
            concave_mean = float(request.form.get("concave"))
            symmetry_mean = float(request.form.get("symmetry"))
            fractal_dimension_mean = float(request.form.get("fractal"))

            # Agrupa os dados em um array para predição
            features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                  compactness_mean, concavity_mean, concave_mean, symmetry_mean, fractal_dimension_mean]])
            
            # Realiza a predição
            previsao = modelo.predict(features)

            # Renderiza a página com o resultado
            return render_template("index.html", previsao=previsao[0])
        
        except Exception as e:
            return render_template("index.html", previsao=f"Erro: {e}")

    # Se for GET, apenas renderiza o formulário
    return render_template("index.html")


 #Rota para fazer previsão
# @app.route('/predict', methods=["POST"])
# def predict():
#     dados = request.get_json()
#     if not dados:
#             return jsonify({'erro': 'Nenhum dado recebido'}), 400

#     try:
#         # Espera uma lista de 30 features como input
#             entrada = np.array(dados['features']).reshape(1, -1)
#             predicao = modelo.predict(entrada)
#             resultado = 'Maligno' if predicao[0] == 1 else 'Benigno'
#             return jsonify({'resultado': resultado})
#     except Exception as e:
#         return jsonify({'erro': str(e)}), 500

# Executa o app
if __name__ == '__main__':
    app.run(debug=True)