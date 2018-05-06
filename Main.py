from flask import Flask,jsonify, request

from Model.Dataset import tasks

from Model.Dataset import Iris

from flasgger import Swagger

from sklearn.externals import joblib

import numpy as np

app = Flask(__name__)
Swagger(app)

@app.route('/input/task', methods=['POST'])
def predict():
    """

    Ini Adalah Endpoint Untuk Memprediksi Makanan

    ---

    tags:

        - Rest Controller

    parameters:

      - name: body

        in: body

        required: true

        schema:

          id: Calories

          required:

            - Calories

            - Fat

          properties:

            Calories:

              type: int

              description: Please input with valid Calories.

              default: 0

            Fat:

              type: int

              description: Please input with valid Fat.

              default: 0

    responses:

        200:

            description: Success Input

    """
    new_task = request.get_json()

    calories = new_task['Calories']
    fat = new_task['Fat']

    X_New = np.array([[calories,fat]])

    clf = joblib.load('static/Caloria.pkl')

    resultPredict = clf[0].predict(X_New)

    return jsonify({'message' : str(resultPredict)})
    # return jsonify({'message' : format(clf[1].target_names[resultPredict])})

app.run(debug=True)

