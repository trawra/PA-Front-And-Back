import sys
from pprint import pprint
from sklearn.neural_network import MLPClassifier
import string
import numpy as np

def vocab():
		return list(string.ascii_lowercase + string.ascii_uppercase + "_")

def getNums(vocab, sentence):
		nums = []
		for word in vocab:
				nums.append(sentence.count(word))
		return nums

def get_model():

	pckg_list = sys.modules
	mods = []

	for i in pckg_list.keys():
		try:
			mods.extend(dir(pckg_list[i]))
		except:
			pass
	mods = list(set(mods))
	del(pckg_list)

	vocabulary = vocab()
	print("Indexed Modules")

	nums = []

	for i in mods:
		nums.append(np.array(getNums(vocabulary, i)))

	print("Training Model . . .")
	nums = np.array(nums)
	model = MLPClassifier(solver = 'adam', hidden_layer_sizes=(128, 128), max_iter = 10000)
	model.fit(nums, mods)
	print("Finished Training!")


	return model

if __name__ == "__main__":
	m = get_model()
	n = getNums(vocab(), "pi")
	print(m.predict([n]))