# DeepRetrieval
Deep Feature based image retrieval library.


# SETUP Instructions

These steps assume that the nessesary model files have been downloaded into the src/ directory.

Execute the following instructions to generate the index.

1. python ImageRetrieval.py --dataset_folder <local_dataset_folder> --create_index --generate_image_index

2. python app.py

If you change the local dataset folder of the files within please remember to reindex. It is not nesseary to reindex once you move application to another machine. Provided the file structure stays the same within the database folder.

# Index files

To add index files for display place them in a folder named static within the src directory. The directory will be used for reading images when the closest match is found. This directory should not be modified without reindexing after. All files within the index need to be present within this folder. The program will check for the indexed files within this directory even if the indexing was done on another machine with a different folder structure. The application looks for only the class name within the folder you have created in the static folder.

