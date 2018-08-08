import model
from flask import Flask, render_template, json, request

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/<page_name>/')
def render_static(page_name):
    return render_template('%s' %page_name)


@app.route('/postjson', methods=['POST'])
def postJson():
    print (request.is_json)
    data = request.get_json()
    print (data)

    data['22'] = data['22'][0] - data['22'][1]
    data['25'] = data['25'][0] - data['25'][1]
    temp = 0
    for i in range(5):
        if data['24'][i] == (100-47-7-i*7):
            temp += 1
    data['24'] = temp
    data['total_recall'] = data['22'] + data['25']

    print (data)


    return  'JSON DONE'


@app.route('/classifier', methods=['GET', 'POST'])
def run_classifier():
    print (request.is_json)
    data = request.get_json()
    return model.api_predict(data)

# start python
if __name__ == "__main__":
    app.run(debug=True)
