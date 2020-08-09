import os
import modify_photo as modp
import manager_of_path


def manage_image(mp):
    total_classes = 500
    train_classes = 400  # 166
    validation_classes = 100
    i = 0
    j = 0
    os.makedirs(mp.path_train_original, exist_ok=True)
    os.makedirs(mp.path_validation_modified, exist_ok=True)
    os.makedirs(mp.path_train_modified, exist_ok=True)
    os.makedirs(mp.path_validation_original, exist_ok=True)
    while i < train_classes:
        name = str(i).zfill(4)
        path_image = mp.get_image_path(name)
        file = os.listdir(path_image)
        list = modp.open_cv2(path_image, file)
        j_modified = modp.blur(list, j, mp.path_train_modified)
        j_original = modp.not_modified(list, j, mp.path_train_original)
        if j_original != j_modified:
            print("error image processing")
        j = j_original
        print(path_image)
        print(file)
        i = i + 1
    j = 0

    while i < validation_classes + train_classes:
        name = str(i).zfill(4)
        path_image = mp.get_image_path(name)
        file = os.listdir(path_image)
        list = modp.open_cv2(path_image, file)
        j_modified = modp.blur(list, j, mp.path_validation_modified)
        j_original = modp.not_modified(list, j, mp.path_validation_original)
        if j_original != j_modified:
            print("error image processing")
        j = j_original
        print(path_image)
        print(file)
        i = i + 1


path = "/home/bernabei/carla0.8.4/PythonClient/_out"
mp = manager_of_path.ManagerOfPath(path)
manage_image(mp)
