import os


def copy_yolo_tags_with_a_new_tag():
    g = os.walk("D:/_data/zyb_data_frame")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if os.path.splitext(file_name)[1] != '.txt':
                continue
            rf = open(os.path.join(path, file_name), "r", encoding='UTF-8')
            lines = rf.readlines()
            new_lines = []
            for l in lines:
                words = l.split(' ')
                if words[0] == '3' or words[0] == '4' or words[0] == '5':
                    words[0] = '2'
                    new_lines.append(" ".join(words))
                elif words[0] == '7' or words[0] == '8' or words[0] == '9' or words[0] == '10' or words[0] == '11':
                    words[0] = '6'
                    new_lines.append(" ".join(words))

            lines.extend(new_lines)
            print(len(new_lines), len(lines), new_lines)
            rf.close()
            wf = open(os.path.join(path, file_name), "w+")
            wf.writelines(lines)
            wf.close()


copy_yolo_tags_with_a_new_tag()
