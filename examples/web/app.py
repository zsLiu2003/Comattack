# app.py

from flask import Flask, render_template, abort
from products import products

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", products=products)

@app.route("/product/<int:product_id>")
def product_detail(product_id):
    if 0 <= product_id < len(products):
        return render_template("product_detail.html", product=products[product_id])
    else:
        abort(404)

if __name__ == "__main__":
    app.run(debug=True)