import numpy as np
from config import get_args

def read_txt(cfg):
    inputpath = './Koeppen-Geiger-ASCII.txt'
    # s_namedict = {"Af": 1, "Am": 2, "As": 3, "Aw": 4, "BWk": 5, "BWh": 6,
    #               "BSk": 7, "BSh": 8, "Cfa": 9, "Cfb": 10, "Cfc": 11, "Csa": 12,
    #               "Csb": 13, "Csc": 14, "Cwa": 15, "Cwb": 16, "Cwc": 17, "Dfa": 18,
    #               "Dfb": 19, "Dfc": 20, "Dfd": 21, "Dsa": 22, "Dsb": 23, "Dsc": 24,
    #               "Dsd": 25, "Dwa": 26, "Dwb": 27, "Dwc": 28, "Dwd": 29, "EF": 30, "ET": 31}
    s_namedict = {"Af": 1, "Am": 1, "As": 1, "Aw": 1, "BWk": 2, "BWh": 2,
                  "BSk": 2, "BSh": 2, "Cfa": 3, "Cfb": 3, "Cfc": 3, "Csa": 3,
                  "Csb": 3, "Csc": 3, "Cwa": 3, "Cwb": 3, "Cwc": 3, "Dfa": 4,
                  "Dfb": 4, "Dfc": 4, "Dfd": 4, "Dsa": 4, "Dsb": 4, "Dsc": 4,
                  "Dsd": 4, "Dwa": 4, "Dwb": 4, "Dwc": 4, "Dwd": 4, "EF": 5, "ET": 5}
    with open(inputpath, 'r', encoding='utf-8') as infile:
        mask = np.zeros((360,720))
        a = np.linspace(-89.75, 89.75, 360)
        b = np.linspace(-179.75, 179.75, 720)
        # 第二种：每行分开读取

        for line in infile:
            data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
            print(data_line)
            ac=np.where(a == float(data_line[0]))
            bc = np.where(b==float(data_line[1]))
            mask[ac,bc] = s_namedict[data_line[2]]
        PATH = cfg['inputs_path'] + cfg['product'] + '/' + str(cfg["spatial_resolution"]) + '/'+str(cfg['selected_year'][0])+'/'
        mask = np.flipud(mask)
        # np.savetxt("mask_0dot5.csv", mask, delimiter=",")
        np.save(PATH+"climates_mask.npy",mask)
        # print(data2)
            # 输出：[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        np.save(PATH+"lon_climates.npy",b)
        np.save(PATH+"lat_climates.npy",a)






if __name__ == "__main__":

    cfg = get_args()
    read_txt(cfg)

