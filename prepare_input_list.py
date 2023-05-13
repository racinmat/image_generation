import os
import os.path as osp
import glob
# images_path = 'animefest AI original'
images_path = 'animefest_selected_bench'
# images_path = 'animefest_selected_bench_small'
extension = 'png'
images = glob.glob(f"{images_path}/**/*.{extension}") + glob.glob(f"{images_path}/*.{extension}")
cur_dir = os.getcwd()
whole_paths = [osp.join(cur_dir, i) for i in images]
# output_file = 'image_whole_list.txt'
output_file = 'bench_list.txt'
# output_file = 'bench_list_small.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for i in whole_paths:
        f.write(f'{i}\n')
print('done')
