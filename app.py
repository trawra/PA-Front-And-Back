from flask import Flask
from flask_restful import Api, reqparse, Resource
from helper_mod import get_model, vocab, getNums

# Get ML Model
model = get_model()
voc = vocab()

app = Flask(app)
api = Api(app)

data_parser = reqparse.RequestParser()
data_parser.add_argument("word", type = str, help = "Word", required = True)

class CodeCompleter(Resource):

	def get(self):
		args = data_parser.parse_args()
		nums = getNums(voc, args["word"])
		ret_list = model.predict([nums])[0]
		