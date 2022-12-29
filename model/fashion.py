from try_on import fit, pre_load
from flask import Flask, request
import requests

app = Flask(__name__)

pre_load()

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        hf = request.files['human_img']
        cf = request.files['cloth_img']
        human_file_path = "data/input/" + hf.filename
        cloth_file_path = "data/input/" + cf.filename
        hf.save(human_file_path)
        cf.save(cloth_file_path)

        pair_list = open("data/input/test_pairs.txt", "w")
        pair = hf.filename + " " + cf.filename
        pair_list.write(pair)
        pair_list.close()

        fit()

        f = open("data/output/try-on/" + hf.filename)

        url = 'http://52.79.134.43:5000/receive?id='+hf.filename.replace('.jpg', '')
        upload = {'image': open(f.name, 'rb')}
        res = requests.post(url, files=upload)
        f.close()

    return "success"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)