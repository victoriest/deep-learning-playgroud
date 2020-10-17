char_list = '0123456789'
char_list = char_list + '^'
num_of_classes = len(char_list)

rnn_unit = 256

img_h = 32
img_w = 170
batch_size = 64
max_label_length = 10

char_to_id = {j: i for i, j in enumerate(char_list)}
id_to_char = {i: j for i, j in enumerate(char_list)}
