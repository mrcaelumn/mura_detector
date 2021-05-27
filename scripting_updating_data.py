import os

name_dir = "mura_data/test_data"
for i, filename in enumerate(os.listdir(name_dir)):
    print(i, filename)
    dst = "normal_" + filename + ".bmp"
    src = name_dir + "/" + filename
    dst = name_dir + "/" + dst
    print(src, dst)
    os.rename(src, dst)
    # os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".bmp")