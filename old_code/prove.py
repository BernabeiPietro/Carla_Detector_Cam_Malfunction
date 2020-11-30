import os


def open_image(name_folder):
    files = []
# r=root, d=directories, f = files
    for r, d, f in os.walk("./"+name_folder):
        for file in f:
            if '.png' or '.jpg' in file:
                files.append(os.path.join(r, file))

    return files