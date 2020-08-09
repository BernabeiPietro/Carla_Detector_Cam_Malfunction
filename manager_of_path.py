class ManagerOfPath:

    path_camera_RGB = "/CameraRGB"
    path_modified = "/modified/"
    path_original = "/orginal/"
    path_episode="/episode_"
    path_validation = "/validation"
    path_train = "/train"

    def __init__(self,path_upper_folder_image):
        path_of_classes = path_upper_folder_image
        self.path_episode_long = path_of_classes + self.path_episode

        self.path_train_modified = path_of_classes + self.path_train + self.path_modified
        self.path_train_original = path_of_classes + self.path_train + self.path_original
        self.path_validation_modified = path_of_classes + self.path_validation + self.path_modified
        self.path_validation_original = path_of_classes + self.path_validation + self.path_original

    def get_image_path(self,name):
        return self.path_episode_long + name + self.path_camera_RGB