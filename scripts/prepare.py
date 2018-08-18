import os
from numpy import array
from pickle import load
import collections
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def getImgIds_ByFolder(folderPath):
    imgIds = []
    for filename in os.listdir(folderPath):
        imgIds.append(os.path.splitext(filename)[0]) # remove exstantion
    return imgIds

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load clean descriptions into memory
def load_clean_descriptions(filename, vocab_size):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line
		tokens = line.split(',')
		if len(line) < 2:
			continue
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# create list
		if image_id not in descriptions:
			descriptions[image_id] = list()
		# wrap description in tokens
		desc = str(vocab_size-2) + ',' + ','.join(image_desc) + ','+str(vocab_size-1)
		# store
		descriptions[image_id].append(desc)
	return descriptions


# create sequences of images, input sequences and one-hot output
def create_sequences(descriptions, vocab_size, photosFeatures):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = desc.split(',')
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=4)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photosFeatures[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

# +1 for padding '0'
# +2 for start/stop sequence tags
vocab_size = 77 + 1 + 2

train = getImgIds_ByFolder('D:/DATASET/VRT/train')

train_features = load_photo_features('features.pkl', train)
print('Photos: %d' % len(train_features))

desc_map = load_clean_descriptions('descriptions.csv', vocab_size)

X1,X2,y = create_sequences(desc_map, vocab_size, train_features)
print (X2)
print (y)
