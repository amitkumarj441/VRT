from os import listdir
from pickle import dump
from keras.preprocessing.image import load_img

fout = open('valid_img_files.csv','w', encoding='utf-8')

def extract_valid_filenames(directory):
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		filename_for_move = directory + '/corrupted/' + name
		try:
			image = load_img(filename, target_size=(224, 224))
		except Exception:
			print('>Corrupted file: %s' % filename)
		else:
			fout.write('%s\n' % (filename))
	return


directory = 'D:/AMIT/Dataset/VRT/train'
extract_valid_filenames(directory)

fout.close()
