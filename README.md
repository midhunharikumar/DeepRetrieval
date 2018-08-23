# DeepRetrieval
Deep Feature based image retrieval library.


# SETUP Instructions

These steps assume that the nessesary model files have been downloaded into the src/ directory.

Execute the following instructions to generate the index.

1. python ImageRetrieval.py --dataset_folder <local_dataset_folder> --create_index --generate_image_index

2. python App.py

If you change the local dataset folder of the files within please remember to reindex. It is not nesseary to reindex once you move application to another machine. Provided the file structure stays the same within the database folder.

