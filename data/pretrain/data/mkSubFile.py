#ecoding=utf-8
import os
import time
#将大的csv文件拆分多个小的csv文件

def mkSubFile(lines, head, srcName, sub):
    [des_filename, extname] = os.path.splitext(srcName)
    filename = des_filename + '_' + str(sub) + extname
    print('make file: %s' % filename)
    fout = open(filename, 'w')
    try:
        fout.writelines([head])
        fout.writelines(lines)
        return sub + 1
    finally:
        fout.close()


def splitByLineCount(filename, count):
    fin = open(filename,encoding="utf-8")
    try:
        head = fin.readline()
        buf = []
        sub = 1
        for line in fin:
            buf.append(line)
            if len(buf) == count:
                sub = mkSubFile(buf, head, filename, sub)
                buf = []
                print(count)
        if len(buf) != 0:
            sub = mkSubFile(buf, head, filename, sub)
    finally:
        fin.close()


if __name__ == '__main__':
    begin = time.time()
    splitByLineCount("now.csv", 1000000)#每个小的csv文件存放
    end = time.time()
    print('time is %d seconds ' % (end - begin))