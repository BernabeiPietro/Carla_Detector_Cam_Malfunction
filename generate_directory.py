from manager_of_path import ManagerOfPath

if __name__ == '__main__':
    path = "/home/bernabei/carla0.8.4/PythonClient/_out/"
    classes_of_modified = ["blur", "black", "brightness", "200_death_pixels", "nodemos", "noise", "sharpness",
                           "brokenlens", "icelens", "banding", "greyscale", "50_death_pixels", "condensation",
                           "dirty_lens", "chromaticaberration", "rain", "all"]
    mp=ManagerOfPath(path,classes_of_modified,True);
    mp.generate_every_path(classes_of_modified);