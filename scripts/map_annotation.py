import os
from pickle import load
import collections

def create_map_id_to_word():
    map_id_to_word = collections.OrderedDict()

    # Collect classes
    f = open('D:/AMIT/Dataset/VRT/Metadata/challenge-2018-classes-vrd.csv','r', encoding='utf-8')
    classes_lines = f.read().split('\n') # "\r\n" if needed
    f.close()
    for line in classes_lines:
        if line != "": # add other needed checks to skip titles
            words = line.split(',')
            map_id_to_word[words[0]] = words[1]
    print ('Classes Nb: %d' % (len(map_id_to_word)))

    # collect attributes
    f = open('D:/AMIT/Dataset/VRT/Metadata/challenge-2018-attributes-description.csv','r', encoding='utf-8')
    attr_lines = f.read().split('\n') # "\r\n" if needed
    f.close()
    for line in attr_lines:
        if line != "": # add other needed checks to skip titles
            words = line.split(',')
            map_id_to_word[words[0]] = words[1]
    print ('Classes And Attributes Nb: %d' % (len(map_id_to_word)))

    # collect relations 
    f = open('D:/AMIT/Dataset/VRT/Metadata/challenge-2018-relationships-description.csv','r', encoding='utf-8')
    rel_lines = f.read().split('\n') # "\r\n" if needed
    f.close()
    for line in rel_lines:
        if line != "": # add other needed checks to skip titles
            words = line.split(',')
            map_id_to_word[words[0]] = words[1]
    print ('Classes, Attributes And Relations Nb: %d' % (len(map_id_to_word)))

    return map_id_to_word

def save_descriptions(annot_fname, map_id_to_word):
    f = open(annot_fname,'r', encoding='utf-8')
    lines = f.read().split('\n') # "\r\n" if needed
    f.close()

    # zero-based mapper ids to indices
    map_id_to_index = list(map_id_to_word.keys())

    filesNotFoundNb = 0
    fout = open('descriptions.csv','w', encoding='utf-8')
    for line in lines:
        if line != "": # add other needed checks to skip titles
            # ImageID,ThumbnailURL,Rotation
            words = line.split(',')

            # Ensure that such file exist
            fname = 'D:/AMIT/Dataset/VRT/train/' + words[0] + '.jpg'
            if os.path.isfile(fname) == True:
                fout.write('%s,%d,%d,%d\n' % (words[0], 1 + map_id_to_index.index(words[1]), 1 + map_id_to_index.index(words[2]), 1 + map_id_to_index.index(words[11]) ))
            else:
                filesNotFoundNb = filesNotFoundNb + 1
    fout.close()
    if filesNotFoundNb > 0:
        print ('During descriptions forming were not found %d images' % (filesNotFoundNb))

map_id_to_word = create_map_id_to_word()

save_descriptions('annotations.csv', map_id_to_word)
