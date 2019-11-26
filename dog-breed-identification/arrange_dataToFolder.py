import os
from glob import glob


path = os.getcwd()
main_folder = glob(path + "/data/train_2/*/" + "*.*")
# all_images = os.listdir(main_folder)
new_path = path + "/data/train/"

for i in main_folder:
    filename = i.split('/')[-1]
    new_loc = new_path + filename
    comm = "mv " + i + " " + new_loc
    os.system(comm)

