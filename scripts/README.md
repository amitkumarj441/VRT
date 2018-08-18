# SCRIPTS

Creating list of thumbnail files - `thumbanail.py`

Downloadi images according to thumbnail list. Images count could be controlled - `download.py`

Filter corrupt images (actually it provide list and we have to delete files manually) - `filter_images.py` 

Based on the downloaded list of files prepare features by specific NN(VGG) and output layer(FF, or last CNN) - `prepare_images.py` 

Copy annotations according to existing list of images - `annotations.py`

Maping annotations to indexed representation - `map_annotation.py`
