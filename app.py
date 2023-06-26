from flask import Flask, request, jsonify
from K_Means import K_Means
import numpy as np



app = Flask(__name__)

@app.route('/data', methods=['POST'])
def get_data():
    data = request.get_json()
    print(data)  
    get_number=data.get("number")

    
    return jsonify(K_Means(get_number))

if __name__ == '__main__':
    app.run()