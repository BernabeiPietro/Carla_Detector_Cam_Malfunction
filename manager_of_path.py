import os


class ManagerOfPath:

    path_camera_RGB = "/CameraRGB"
    path_modified = "/modified/"
    path_original = "/orginal/"
    path_episode="/episode_"
    path_validation = "/validation"
    path_train = "/train"
    path_classes_modify={}

    def __init__(self,path_upper_folder_image,classes_of_modified,type_of_folder=True):
        #se True, definisce n_percorsi differenti quante le classi di modfiche, se False, definisce un unico percorso per tutte le classi di modifiche
        path_of_classes = path_upper_folder_image
        self.setting_type_folder=type_of_folder
        if type_of_folder:
            for c in classes_of_modified:
                self.path_classes_modify[c]=path_of_classes+"/"+c
        else:
            self.path_classes_modify["all"]=path_of_classes+"/all"
        self.path_episode_long = path_of_classes + self.path_episode

    def get_image_path(self,name):
        return self.path_episode_long + name + self.path_camera_RGB

    def get_path_classes(self,classes):
        dict_path={}
        if(len(self.path_classes_modify.keys())==1):
            classes="all"
        dict_path["train_original"] = self.path_classes_modify[classes] + self.path_train + self.path_original
        dict_path["train_modified"] = self.path_classes_modify[classes] +  self.path_train + self.path_modified
        dict_path["validation_original"] = self.path_classes_modify[
                                               classes] +  self.path_validation + self.path_original
        dict_path["validation_modified"] = self.path_classes_modify[
                                               classes] +  self.path_validation + self.path_modified
        for path in dict_path:
            os.makedirs(dict_path[path], exist_ok=True)

        return dict_path