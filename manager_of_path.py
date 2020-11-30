import os


def open_image(name_folder):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk("./" + name_folder):
        for file in f:
            if '.png' or '.jpg' in file:
                files.append(os.path.join(r, file))

    return files


class ManagerOfPath:
    path_camera_RGB = "/CameraRGB"
    path_modified = "/modified/"
    path_original = "/orginal/"
    path_episode = "/episode_"
    path_validation = "/validation"
    path_train = "/train"
    path_test = "/test"
    path_test_all = "/test_all"
    path_classes_modify = {}
    path_checkpoint = "/checkpoint/"

    def __init__(self, path_upper_folder_image, classes_of_modified, type_of_folder=True):
        # se True, definisce n_percorsi differenti quante le classi di modfiche, se False, definisce un unico percorso per tutte le classi di modifiche
        self.path_of_classes = path_upper_folder_image
        self.setting_type_folder = type_of_folder
        if type_of_folder:
            for c in classes_of_modified:
                self.path_classes_modify[c] = self.path_of_classes + c
        else:
            self.path_classes_modify["all"] = self.path_of_classes + "/all"
        self.path_episode_long = self.path_of_classes + self.path_episode

    def get_image_path(self, name):
        return self.path_episode_long + name + self.path_camera_RGB

    def get_path_classes(self, classes):
        dict_path = {}
        if (not (self.setting_type_folder)):
            classes = "all"
        dict_path["train_original"] = self.path_classes_modify[classes] + self.path_train + self.path_original
        dict_path["train_modified"] = self.path_classes_modify[classes] + self.path_train + self.path_modified
        dict_path["validation_original"] = self.path_classes_modify[
                                               classes] + self.path_validation + self.path_original
        dict_path["validation_modified"] = self.path_classes_modify[
                                               classes] + self.path_validation + self.path_modified
        dict_path["test_original"] = self.path_classes_modify[classes] + self.path_test + self.path_original
        dict_path["test_modified"] = self.path_classes_modify[classes] + self.path_test + self.path_modified
        dict_path["test_all_original"] = self.path_classes_modify[classes] + self.path_test_all + self.path_original
        dict_path["test_all_modified"] = self.path_classes_modify[classes] + self.path_test_all + self.path_modified
        dict_path["checkpoint"] = self.path_classes_modify[classes] + self.path_checkpoint
        dict_path["train"] = self.path_classes_modify[classes] + self.path_train
        dict_path["validation"] = self.path_classes_modify[classes] + self.path_validation
        dict_path["test"] = self.path_classes_modify[classes] + self.path_test
        dict_path["test_all"] = self.path_classes_modify[classes] + self.path_test_all
        for path in dict_path:
            os.makedirs(dict_path[path], exist_ok=True)
        return dict_path

    def generate_every_path(self, classes):
        for cl in classes:
            self.get_path_classes(cl)
