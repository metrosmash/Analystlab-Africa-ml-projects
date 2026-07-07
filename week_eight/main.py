from flask import Flask
from flask import request
from markupsafe import escape 
from flask import render_template
from safetensors.numpy import save_file, load_file
import logging
import joblib
import os
from sklearn.linear_model import LogisticRegression


# configure logging 
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Default Character constants
DEFAULT_EMBARKED = 'Southampton'
DEFAULT_FARE = 33
DEFAULT_AGE = 30
DEFAULT_GENDER = 'Female'
DEFAULT_TITLE = 'Mrs.'
DEFAULT_CLASS = 'Second'
DEFAULT_CABIN = 'C'
DEFAULT_SIBSP = 0
DEFAULT_PARCH = 0

# Get absolute path to project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'assets', 'logreg.safetensors')

def load_model():
    try:
        # loading the model 
        loaded_tensors = load_file(MODEL_DIR)

        revived_model = LogisticRegression()

        revived_model.classes_ = loaded_tensors["classes"]
        revived_model.coef_ = loaded_tensors ["coef"]
        revived_model.intercept_ = loaded_tensors["intercept"]

        logger.info("## Model loaded sucessfully")
        return revived_model
    except FileNotFoundError as e:
        logger.error(f"## Failed to load the model : {e}")
        logger.error("make sure the model is in the right folder")

with app.app_context():
        model = load_model()


@app.route("/", methods = ['POST','GET'])
def get_user_input():

    if request.method == 'POST':
        inputed_embarked = request.form['inputed_embarked']
        inputed_Age = request.form['inputed_Age']
        inputed_gender = request.form['inputed_gender']
        inputed_title = request.form['inputed_title']
        inputed_class = request.form['inputed_class']
        inputed_sibsp = request.form['inputed_sibsp']
        inputed_parch = request.form['inputed_parch']

        # core needed integers 
        Age = int(inputed_Age)
        is_female = 1 if inputed_gender == "Female" else 0
        sibsp = int(inputed_sibsp)
        parch = int(inputed_parch)

        # is_alone column 
        is_alone = 0 if sibsp == 0 and parch == 0 else 1 
        family_size = sibsp + parch + 1
        family_bin_1 = 1 if family_size >= 2 and family_size <=3 else 0
        family_bin_2 = 1 if family_size >= 4 else 0

        # Port of Embarkation 
        Embarked_Q       = 0
        Embarked_S       = 0
        if inputed_embarked[0] == 'Q':
            Embarked_Q = 1
        elif inputed_embarked[0] == 'S':
            Embarked_S = 1

        # Ship Class
        Pclass_2 = 0
        Pclass_3  = 0
        if inputed_class == 'Second':
            Pclass_2 = 1
        elif inputed_class == 'Third':
            Pclass_3 = 1

        # Title
        title_Master  = 0
        title_Miss    = 0
        title_Mr      = 0
        title_Mrs     = 0
        title_Rev     = 0
        title_Mlle    = 0
        title_Unknown = 0
        title_nan     = 0
        title_map = {
            'Master': 'title_Master',
            'Miss':   'title_Miss',
            'Mr':     'title_Mr',
            'Mlle' : 'title_Mlle',
            'Mrs':    'title_Mrs',
            'Rev':    'title_Rev',
            'Unknown': 'title_Unknown',
        }
        if inputed_title in title_map:
            locals()[title_map[inputed_title]]  # resolve name
        # Use explicit assignment to avoid locals() mutation issues
        if inputed_title == 'Master': title_Master = 1
        elif inputed_title == 'Miss': title_Miss = 1
        elif inputed_title == 'Mr':   title_Mr = 1
        elif inputed_title == 'Mlle' : title_Mlle = 1
        elif inputed_title == 'Mrs':  title_Mrs = 1
        elif inputed_title == 'Rev':  title_Rev = 1
        elif inputed_title == 'Unknown': title_Unknown = 1

        # Build feature vector matching training data format
        user_passenger = [[
            Age, is_female, is_alone, Embarked_Q, Embarked_S,
            title_Master, title_Miss,title_Mlle, title_Mr, title_Mrs, title_Rev,
            title_Unknown,family_bin_1,family_bin_2, Pclass_2, Pclass_3, 
        ]]

        # work on this
        Y_pred = model.predict_proba(user_passenger)
        survival_pct = Y_pred[0][1] * 100
        logger.info(survival_pct)
        model_output = f'Your Character has {survival_pct:.1f}% Chance of Surviving!'
        return render_template("index.html",
            model_output = model_output,
            inputed_Age = inputed_Age,
            inputed_gender = inputed_gender,
            inputed_title =inputed_title,
            inputed_embarked = inputed_embarked,
            inputed_class = inputed_class,
            inputed_sibsp = inputed_sibsp,
            inputed_parch = inputed_parch,) 
    else:
        return render_template("index.html",
            model_output = " ",
            inputed_Age = DEFAULT_AGE,
            inputed_embarked = DEFAULT_EMBARKED,           
            inputed_gender=DEFAULT_GENDER,
            inputed_title=DEFAULT_TITLE,
            inputed_class=DEFAULT_CLASS,
            inputed_sibsp=DEFAULT_SIBSP,
            inputed_parch=DEFAULT_PARCH)


if __name__ == '__main__':
    logger.info("Titanic flask api Starting")

    app.run(
        debug = False
    )
