import os
import sys
import urllib
import urllib.request
import shutil
import os.path

f = open('thumburls.csv','r', encoding='utf-8')
thumb_lines = f.read().split('\n') # "\r\n" if needed
f.close()

thumb_urls = {}
for line in thumb_lines[1:]:
    if line != "": # add other needed checks to skip titles
        # ImageID,ThumbnailURL,Rotation
        words = line.split(',')
        thumb_urls[words[0]] = words[1]



f = open('D:/AMIT/Dataset/VRT/Annotations/challenge-2018-train-vrd.csv','r', encoding='utf-8')
vrd_lines = f.read().split('\n') # "\r\n" if needed
f.close()

# download files which had not meen dowloaded before
for line in vrd_lines[1:27000]:
    if line != "": # add other needed checks to skip titles
        # ImageID,LabelName1,LabelName2,XMin1,XMax1,YMin1,YMax1,XMin2,XMax2,YMin2,YMax2,RelationshipLabel
        words = line.split(',')
        img_id = words[0]
        fname = 'D:/DATASET/VRT/train/' + img_id + '.jpg'
        if os.path.isfile(fname) == False:
            try:
                img_url = thumb_urls[img_id]
                with urllib.request.urlopen(img_url) as response, open(fname, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            except:
                print("Unexpected error:", sys.exc_info()[0])
