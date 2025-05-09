from flask import Flask, render_template, request, send_file, jsonify
from c45 import C45
from sf_pd import SF_PD
from kp import KP
from gizi import GIZI

app = Flask(__name__)

c45 = C45("dataset/dataset_stunting.csv")
c45.load_data()
c45.kfold_validation()  

sf = SF_PD("dataset/breast-cancer.csv")
sf.load_data()
sf.select_features()
feature_scores = sf.get_feature_probabilities().to_dict(orient='records')
selected_features = sf.get_selected_features().to_dict(orient='records')

kp = KP("dataset/dataset_kp.csv")
kp.load_data()
kp.kfold_validation()

gizi = GIZI("dataset/dataset_gizi.csv")
gizi.load_data()
gizi.kfold_validation()



@app.route("/")
def welcome():
    return render_template("index.html")

@app.route('/index_c45')
def home_c45():
    return render_template('index_c45.html')

@app.route('/informasi_c45')
def informasi_c45():
    results, best_fold, best_metrics, best_cm_path = c45.get_results()
    kfold_aggregates = c45.get_kfold_aggregates()
    rules = c45.get_rules()
    return render_template('informasi_c45.html', 
                           results=results, 
                           best_fold=best_fold, 
                           best_metrics=best_metrics, 
                           best_cm_path=best_cm_path,
                           kfold_aggregates=kfold_aggregates,
                           rules=rules,)

@app.route('/prediksi_c45', methods=['GET', 'POST'])
def prediksi_c45():
    if request.method == 'POST':
        name = request.form['name']
        address = request.form['address']
        parent = request.form['parent']
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])

        prediction = c45.predict(gender, age, weight, height) # ganti `model` dengan nama modelmu
        gender_label = 'Laki-Laki' if gender == 0 else 'Perempuan'
        prediction_label = 'Tidak Stunting' if prediction == 0 else 'Stunting'

        return render_template('prediksi_c45.html',
                               name=name,
                               address=address,
                               parent=parent,
                               gender=gender_label,
                               age=age,
                               weight=weight,
                               height=height,
                               prediction=prediction_label)
    return render_template('prediksi_c45.html')


@app.route('/index_kp')
def home():
    return render_template('index_kp.html')

@app.route('/informasi_kp')
def informasi():
    results, best_fold, best_metrics, best_cm_path = kp.get_results()
    

    # Menghitung rata-rata kfold untuk digunakan di template
    kfold_aggregates = kp.get_kfold_aggregates()

    return render_template('informasi_kp.html', 
                           feature_scores=feature_scores, 
                           selected_features=selected_features, 
                           results=results, 
                           best_fold=best_fold, 
                           best_metrics=best_metrics, 
                           best_cm_path=best_cm_path,
                           kfold_aggregates=kfold_aggregates)

@app.route('/prediksi_kp')
def prediksi():
    return render_template('prediksi_kp.html')

@app.route('/hasil_prediksi_kp', methods=['POST'])
def hasil_prediksi():
    # Ambil data dalam bentuk string
    input_data_str = {feature: request.form[feature] for feature in kp.selected_features}
    
    # Ubah ke float untuk prediksi
    input_data_float = {feature: float(value) for feature, value in input_data_str.items()}
    
    hasil = kp.predict(input_data_float)
    akurasi = kp.get_best_accuracy()
    return render_template('prediksi_kp.html', hasil=hasil, input_data=input_data_str, akurasi=akurasi)

@app.route('/index_gizi')
def home_gizi():
    return render_template('index_gizi.html')

@app.route('/informasi_gizi')
def informasi_gizi():
    results, best_fold, best_metrics, best_cm_path = gizi.get_results()
    # Menghitung rata-rata kfold untuk digunakan di template
    kfold_aggregates = gizi.get_kfold_aggregates()

    return render_template('informasi_gizi.html', 
                           results=results, 
                           best_fold=best_fold, 
                           best_metrics=best_metrics, 
                           best_cm_path=best_cm_path,
                           kfold_aggregates=kfold_aggregates)

@app.route("/prediksi_gizi", methods=["GET", "POST"])
def prediksi_gizi():
    if request.method == "POST":
        age = float(request.form["age"])
        gender = int(request.form["gender"])
        weight = float(request.form["weight"])
        height = float(request.form["height"])

        hasil = gizi.predict(age, gender, weight, height)

        return render_template("prediksi_gizi.html", prediction=hasil, data={
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height
        }, accuracy=gizi.best_accuracy)
    return render_template("prediksi_gizi.html")

if __name__ == "__main__":
        app.run(host='0.0.0.0', port=10000)
