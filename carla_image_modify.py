import os

import manager_of_path
import modify_photo as modp


def manage_image(mp, classes_of_modified, asymmetrical_test, slice_class=slice(None, 16, None)):
    total_episode = 24  # 524
    train_episode = 10  # 400
    validation_episode = 10  # 100
    test_episode = 10  # 24
    i = 0  # numero di episodio da eseguire
    j = 0  # nome dell'ultima  immagine scritto
    if mp.setting_type_folder:  # single class
        indix_of_classes = slice_class
    else:  # all class
        indix_of_classes = 0
        asymmetrical_test = False
    phase = "train"
    while i < train_episode + validation_episode + test_episode:
        if i == train_episode:
            phase = "validation"
            j = 0
        if i == train_episode + validation_episode:
            phase = "test"
            j = 0
        name = str(i).zfill(4)
        path_image = mp.get_image_path(name)
        file = os.listdir(path_image)
        list = modp.open_cv2(path_image, file)
        if mp.setting_type_folder:
            class_of_con = classes_of_modified[indix_of_classes]
        else:
            class_of_con = [classes_of_modified[indix_of_classes]]
        if asymmetrical_test == True and phase == "test":
            princ_class = classes_of_modified[i % len(classes_of_modified)]
            j = modify_photo_test_asimmetrico(class_of_con, mp, j, list, phase, princ_class)
        j = modify_photo(class_of_con, mp, list, j, phase)
        print(path_image)
        print(file)
        i = i + 1
        if not mp.setting_type_folder:
            indix_of_classes = (indix_of_classes + 1) % len(classes_of_modified)


def modify_photo(classes, mp, list_original, j, phase):
    tv_modified = phase + "_modified"
    tv_original = phase + "_original"

    for elem in classes:
        modp.run_methods(elem, list_original, j, mp.get_path_classes(elem)[tv_modified])
        j_original = modp.not_modified(list_original, j, mp.get_path_classes(elem)[tv_original])

    return j_original


def modify_photo_test_asimmetrico(classe_of_con, mp, j, list_original, phase, classe):
    tv_original = phase + "_all" + "_original"
    for elem in classe_of_con:
        if elem == classe:
            tv_path = phase + "_all" + "_modified"
        else:
            tv_path = phase + "_all" + "_original"
        j = modp.run_methods(classe, list_original, j, mp.get_path_classes(elem)[tv_path])
        j = modp.not_modified(list_original, j, mp.get_path_classes(elem)[tv_original])
    return j


path =  "/home/bernabei/carla0.8.4/PythonClient/_out_prima_run/"# "/home/bernabei/carla0.8.4/PythonClient/_out/" # "/home/pietro/Documenti/Unifi/tirocinio/img"
classes_of_modified = ["blur", "black", "brightness", "200_death_pixels", "nodemos", "noise", "sharpness", "brokenlens",
                       "icelens", "banding", "50_death_pixels", "greyscale", "condensation", "dirty_lens",
                       "chromaticaberration", "rain"]
mp = manager_of_path.ManagerOfPath(path, classes_of_modified,True)
manage_image(mp, classes_of_modified, True, slice(None, 1, None))
