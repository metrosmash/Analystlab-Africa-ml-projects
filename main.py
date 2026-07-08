import os
import joblib
import logging
from flask import Flask
from flask import request
from markupsafe import escape 
from flask import render_template
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier


# configure logging 
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Default Character constants 11
DEFAULT_AGE = 30
DEFAULT_GENDER = 'Female'
DEFAULT_CHEST_PAIN_TYPE = 'Atypical Angina'
DEFAULT_RESTING_BP_S = 99
DEFAULT_CHOLESTEROL = 100
DEFAULT_FASTING_BLOOD_SUGAR = 'Above 120mg/dL'
DEFAULT_RESTING_ECG  = 'ST-T Wave Abnormality'
DEFAULT_MAX_HEART_RATE = 150
DEFAULT_EXERCISE_ANGINA = 'Not Present'
DEFAULT_OLDPEAK = 2.2
DEFAULT_ST_SLOPE  = 'Upsloping'

def float_range(start, stop, step):
    while start < stop:
        yield round(start, 2)  # Adjust precision as needed
        start += step


# Get absolute path to project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'assets/model', 'balanced_bagging_rf_model.joblib')

# 
def load_model():
    try:
        # loading the model
        rf_model = joblib.load(MODEL_DIR)

        logger.info("## Model loaded sucessfully")
        return rf_model
    except FileNotFoundError as e:
        logger.error(f"## Failed to load the model : {e}")
        logger.error("make sure the model is in the right folder")

with app.app_context():
        model = load_model()


@app.route("/", methods = ['POST','GET'])
def get_user_input():

    if request.method == 'POST':
        inputed_Age = request.form['inputed_Age']
        inputed_gender = request.form['inputed_gender']
        inputed_cp = request.form['inputed_cp']        
        inputed_rbp = request.form['inputed_rbp']
        inputed_cholestrol = request.form['inputed_cholestrol']
        inputed_fbs = request.form['inputed_fbs']
        inputed_recg = request.form['inputed_recg']
        inputed_mhr = request.form['inputed_mhr']
        inputed_EA = request.form['inputed_EA']
        inputed_Oldpeak = request.form['inputed_Oldpeak']
        inputed_sts = request.form['inputed_sts']   

        # core needed integers 
        Age = int(inputed_Age)
        sex = 0 if inputed_gender == "Female" else 1
        resting_bp_s = float(inputed_rbp)
        cholesterol = float(inputed_cholestrol)
        fasting_blood_sugar  = 1 if inputed_fbs == "Above 120mg/dL" else 0
        max_heart_rate = int(inputed_mhr)
        exercise_angina = 1 if inputed_EA == "Exercise-Induced Angina" else 0 
        oldpeak = float(inputed_Oldpeak)


        # Chest pain type column 
        # when cp_2,_3,_4 = 0  then cp_1 is equal to 1
        cp_2 = 0
        cp_3 = 0
        cp_4 = 0
        if inputed_cp == "Atypical Angina":
            cp_2 = 1
        elif inputed_cp == "Non-Anginal Pain":
            cp_3 = 1
        elif inputed_cp == "Asymptomatic":
            cp_4 = 1

        # Resting ecg column 
        # when ecg_1, _2 = 0  then ecg_0 is equal to 1
        ecg_1 = 0
        ecg_2 = 0
        if inputed_recg == "ST-T Wave Abnormality":
            ecg_1 = 1
        elif inputed_recg == "Left Ventricular Hypertrophy":
            ecg_2 = 1
         
        # ST Slope column 
        # when slope_flat, upsloping = 0 then downsloping is equal to 1 
        slope_flat = 0
        slope_upsloping = 0
        if inputed_sts == "Flat":
            slope_flat = 1
        elif inputed_sts == "Upsloping":
            slope_upsloping = 1

        # Age Bin columns here 
        # when all other bins are 0 40 to 55  is equal to 1 
        age_under_40 = 0
        age_55_65 = 0
        age_over_65 = 0
        if Age < 40:
            age_under_40 = 1
        elif Age > 55 or Age < 65:
            age_55_65 = 1
        elif Age > 65:
            age_over_65 = 1

           

        # Build feature vector matching training data format
        patients_diagnosis = [[
            Age, sex, resting_bp_s, cholesterol, fasting_blood_sugar,
            max_heart_rate, exercise_angina, oldpeak, cp_2, cp_3, cp_4,
            ecg_1,ecg_2,slope_flat, slope_upsloping, age_55_65,
            age_over_65, age_under_40
        ]]

        # work on this
        Y_pred = model.predict_proba(patients_diagnosis)
        survival_pct = Y_pred[0][1] * 100
        logger.info(survival_pct)
        model_output = f'Patient is: {survival_pct:.1f}% Positive'
        return render_template("index.html",
            model_output = model_output,
            inputed_Age = inputed_Age,
            inputed_gender = inputed_gender,
            inputed_cp =inputed_cp,
            inputed_rbp = inputed_rbp,
            inputed_cholestrol = inputed_cholestrol,
            inputed_fbs = inputed_fbs,
            inputed_recg = inputed_recg,
            inputed_mhr = inputed_mhr,
            inputed_EA = inputed_EA,
            inputed_Oldpeak = inputed_Oldpeak,
            inputed_sts = inputed_sts,
            float_range = float_range) 
    else:
        return render_template("index.html",
            model_output = " ",
            inputed_Age = DEFAULT_AGE,           
            inputed_gender=DEFAULT_GENDER,
            inputed_cp = DEFAULT_CHEST_PAIN_TYPE,
            inputed_rbp = DEFAULT_RESTING_BP_S,
            inputed_cholestrol = DEFAULT_CHOLESTEROL,
            inputed_fbs = DEFAULT_FASTING_BLOOD_SUGAR,
            inputed_recg = DEFAULT_RESTING_ECG,
            inputed_mhr = DEFAULT_MAX_HEART_RATE,
            inputed_EA = DEFAULT_EXERCISE_ANGINA,
            inputed_Oldpeak = DEFAULT_OLDPEAK,
            inputed_sts = DEFAULT_ST_SLOPE,
            float_range = float_range)


if __name__ == '__main__':
    logger.info("Heart Disease Flask Api Starting")

    app.run(
        debug = False
    )
