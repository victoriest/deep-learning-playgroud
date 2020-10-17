import os

from PIL import Image


def flip_image(file_path, file_name):
    img = Image.open(os.path.join(file_path, file_name))
    img = img.transpose(Image.ROTATE_180)
    dest_name = os.path.join(file_path, os.path.splitext(file_name)[0] + "_180.jpg")
    img.save(dest_name)


def flip_mark(file_path, file_name):
    rf = open(os.path.join(file_path, os.path.splitext(file_name)[0] + ".txt"), "r")
    wf = open(os.path.join(file_path, os.path.splitext(file_name)[0] + "_180.txt"), "w")
    lines = rf.readlines()
    new_lines = []
    for l in lines:
        str_arr = []
        wd = l.split(' ')
        str_arr.append(wd[0])
        str_arr.append(str('%.6f' % (1.0 - float(wd[1]))))
        str_arr.append(str('%.6f' % (1.0 - float(wd[2]))))
        str_arr.append(wd[3])
        str_arr.append(wd[4])
        # for i in range(1, 5):
        #     print(float(wd[i]), (1.0 - float(wd[i])), str('%.6f' % (1.0 - float(wd[i]))))
        #     str_arr.append(str('%.6f' % (1.0 - float(wd[i]))))
        # print(str_arr)
        # str_arr.append("\n")
        new_lines.append(" ".join(str_arr))
    rf.close()
    wf.writelines(new_lines)
    wf.close()


if __name__ == '__main__':
    g = os.walk("E:/_dataset/zyb_stucode_data")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if os.path.splitext(file_name)[1] == '.jpg' or os.path.splitext(file_name)[1] == '.png':
                flip_image(path, file_name)
                flip_mark(path, file_name)
