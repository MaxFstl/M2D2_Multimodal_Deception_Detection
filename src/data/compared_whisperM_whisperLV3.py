# -*- coding: UTF-8 -*-
'''
@Project ：M2D2_Multimodal_Deception_Detection-main 
@File    ：compared_whisperM_whisperLV3.py
@IDE     ：PyCharm 
@Author  ：Winston·H DONG
@Date    ：2026/4/18 14:17 
'''
import os
import filecmp
import shutil
import difflib


def show_file_diff(file1, file2, filename):
    print(f"--- begin compared: {filename} ---")
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        diff = difflib.unified_diff(
            lines1,
            lines2,
            fromfile=file1,
            tofile=file2,
            lineterm=''
        )

        for line in diff:
            print(line)
    print(f"--- file {filename} compared finish ---\n")


def compare_txt_dirs(dir_a, dir_b):

    txts_a = {f for f in os.listdir(dir_a) if f.endswith('.txt')}
    txts_b = {f for f in os.listdir(dir_b) if f.endswith('.txt')}

    common_txts = txts_a & txts_b
    only_in_a = txts_a - txts_b
    only_in_b = txts_b - txts_a

    print("TXT files only in A:", only_in_a)
    print("TXT files only in B:", only_in_b)

    all_equal = True

    for name in sorted(common_txts):
        path_a = os.path.join(dir_a, name)
        path_b = os.path.join(dir_b, name)

        same = filecmp.cmp(path_a, path_b, shallow=False)

        if same:
            print(f"[SAME] {name}")
        else:
            all_equal = False
            print(f"[DIFFERENT] {name}")
            show_file_diff(path_a, path_b, name)

    if all_equal and not only_in_a and not only_in_b:
        print("All txt files with the same name are identical and no extra/missing txt files.")
    else:
        print("compared finished.")


if __name__ == "__main__":
    dir_a = r"../../data/DOLOS/txt_files/txt_M"
    dir_b = r"../../data/DOLOS/txt_files/txt_LV3"
    compare_txt_dirs(dir_a, dir_b)