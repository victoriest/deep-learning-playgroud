import os


def generate_yolov4_manipast():
    f = open('train.txt', 'w')
    g = os.walk("E:/_dataset/zyb_opt_mark_data/yolo")
    lines = []

    for path, dir_list, file_list in g:
        for file_name in file_list:
            # print(os.path.join(path, file_name), os.path.splitext(file_name))
            rf = open(os.path.join(path, file_name), "r")
            lines = rf.readlines()
            new_lines = []
            for l in lines:
                words = l.split(' ')
                words[0] = '7'
                new_lines.append(" ".join(words))
            lines.extend(new_lines)
            print(len(new_lines), len(lines), new_lines)
            rf.close()
            wf = open(os.path.join(path, file_name), "w+")
            wf.writelines(lines)
            wf.close()


generate_yolov4_manipast()