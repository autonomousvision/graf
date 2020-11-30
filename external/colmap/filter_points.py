import os
import sys
import glob
import struct
import numpy as np

def readBinaryPly(pcdFile, fmt='ffffffBBB', fmt_len=27):

    with open(pcdFile, 'rb') as f:
        plyData = f.readlines()

    headLine = plyData.index(b'end_header\n')+1
    plyData = plyData[headLine:]
    plyData = b"".join(plyData)

    n_pts_loaded = int(len(plyData)/fmt_len)

    data = []
    for i in range(n_pts_loaded):
        pts=struct.unpack(fmt, plyData[i*fmt_len:(i+1)*fmt_len])
        data.append(pts)
    data=np.asarray(data)

    return data

def writeBinaryPly(pcdFile, data):
    fmt = '=ffffffBBB'
    fmt_len = 27
    n_pts = data.shape[0]

    with open(pcdFile, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(b'comment\n')
        f.write(b'element vertex %d\n' % n_pts)
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property float nx\n')
        f.write(b'property float ny\n')
        f.write(b'property float nz\n')
        f.write(b'property uchar red\n')
        f.write(b'property uchar green\n')
        f.write(b'property uchar blue\n')
        f.write(b'end_header\n')

        for i in range(n_pts):
            f.write(struct.pack(fmt, *data[i,0:6], *data[i,6:9].astype(np.uint8)))


def filter_ply(object_dir):

    ply_files = sorted(glob.glob(os.path.join(object_dir, 'dense', '*', 'fused.ply')))

    for ply_file in ply_files:
        ply_filter_file = ply_file.replace('.ply', '_filtered.ply')
        plydata = readBinaryPly(ply_file)
        vertex = plydata[:,0:3]
        normal = plydata[:,3:6]
        color = plydata[:,6:9]

        mask = np.mean(color,1)<(0.85 * 255.)
        color = color[mask, :]
        normal = normal[mask, :]
        vertex = vertex[mask, :]
        plydata = np.hstack((vertex, normal, color))
        writeBinaryPly(ply_filter_file, plydata)
        print('Processed file {}'.format(ply_filter_file))

if __name__=='__main__':

    object_dir=sys.argv[1]
    filter_ply(object_dir)
