import os
import psutil


def get_libri_phoneme(fname):
    with open(fname) as file:
        phoneme_list = []
        while(True):
            aline = file.readline()
            if not aline : break
            phoneme_list.append(aline.split("\n")[0])
        
    return phoneme_list

def get_memory_usege():
    pid = os.getpid()
    current_process = psutil.Process(pid)
    current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
    print(f"AFTER  CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

