from flask import Flask, request, render_template
import joblib
import numpy as np

# Cria o app Flask
app = Flask(__name__)

# Carrega o modelo
modelo = joblib.load(r'C:\Users\Lucas Vinicius\Desktop\IA\modelo_rg.pkl')


# Rota padrão
@app.route('/')
def home():
    return 'API do Modelo de Câncer - Online!'

app = Flask(__name__)

@app.route('/home', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Captura os dados do formulário
            radius_mean = float(request.form.get("radius"))
            texture_mean = float(request.form.get("texture"))
            perimeter_mean = float(request.form.get("perimeter"))
            area_mean = float(request.form.get("area"))

            # Agrupa os dados em um array para predição
            features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])
            
            # Realiza a predição
            previsao = modelo.predict(features)

            # Renderiza a página com o resultado
            return render_template("index.html", previsao=previsao[0])
        
        except Exception as e:
            return render_template("index.html", previsao=f"Erro: {e}")

    # Se for GET, apenas renderiza o formulário
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)