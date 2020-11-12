import os
import modify_photo as modp
import manager_of_path


def manage_image(mp, classes_of_modified):
    total_classes = 24#500
    train_classes = 19 # 400
    validation_classes =0 #100
    i = 0
    j = 0
    if mp.setting_type_folder:
        indix_of_classes = len(classes_of_modified)
    else:
        indix_of_classes = 0
    vt = True
    while i < train_classes + validation_classes:
        if i == train_classes:
            vt = False
            j = 0
        name = str(i).zfill(4)
        path_image = mp.get_image_path(name)
        file = os.listdir(path_image)
        list = modp.open_cv2(path_image, file)
        j = modify_photo(classes_of_modified[i%indix_of_classes], mp, list, j, vt,classes_of_modified)
        print(path_image)
        print(file)
        i = i + 1
        if not mp.setting_type_folder:
            indix_of_classes = (indix_of_classes + 1) % len(classes_of_modified)


def modify_photo(classe, mp, list_original, j, tv, classes_of_modified):
    # tv:
    # -true=train
    # -false=validation
    if tv:
        string_tv = "train"
    else:
        string_tv = "validation"
    tv_modified = string_tv + "_modified"
    tv_original = string_tv + "_original"
    # if not mp.setting_type_folder:
    #    j_modified_tot=j*len(classes)-1

    overlap_broken=["broken1.png","broken7.png","broken8.jpg"]

    overlap_ice=["ice.jpg","ice3.png"]

    overlap_banding=["banding1.jpg"]
    list=list_original[:]
    for i in classes_of_modified:
        if i ==classe:
            tv_path=tv_modified
        else:
            tv_path=tv_original

        if "50_death_pixels" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j=j+1
            j = modp.dead_pixel_50(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
        if "200_death_pixels" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.dead_pixel_200(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
        if "blur" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.blur(list, j, mp.get_path_classes(i)[tv_path])
            j=j+1
        if "black" in classe:
            j = modp.black(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            if mp.setting_type_folder:
                j = j

        if "brightness" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.brightness(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "greyscale" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.greyscale(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "nodemos" in classe:
            j = modp.nodemos(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "noise" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.noise(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "sharpness" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.sharpness(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "brokenlens" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.overlap(list, j, mp.get_path_classes(i)[tv_path],mp.path_of_classes+overlap_broken[0],0.35)
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "icelens" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.overlap(list, j, mp.get_path_classes(i)[tv_path],mp.path_of_classes+overlap_ice[0],0.2)
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "banding" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.banding(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "condensation" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.condensation(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "dirty_lens" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.dirty_lens(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "chromaticaberration" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.chromaticaberration(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            if mp.setting_type_folder:
                j = j
        if "rain" in classe:
            j = modp.not_modified(list_original, j, mp.get_path_classes(i)[tv_original])
            j = j + 1
            j = modp.rain(list, j, mp.get_path_classes(i)[tv_path])
            j = j + 1
            if mp.setting_type_folder:
                j = j
    return j



path = "/media/pietro/Volume/Ubuntu/home/pietro/Documenti/Unifi/tirocinio/img/"#"/home/bernabei/carla0.8.4/PythonClient/_out/" #"/home/bernabei/carla0.8.4/PythonClient/_out_prima_run/" #"/home/pietro/Documenti/Unifi/tirocinio/img"
classes_of_modified = ["blur", "black", "brightness",  "200_death_pixels","nodemos","noise","sharpness","brokenlens","icelens","banding","50_death_pixels","greyscale","condensation","dirty_lens","chromaticaberration","rain"]
mp = manager_of_path.ManagerOfPath(path, classes_of_modified, True)
manage_image(mp, classes_of_modified)




