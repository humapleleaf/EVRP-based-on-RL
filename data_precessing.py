import matplotlib.pyplot as plt
import os
from math import radians, cos, sin, asin, sqrt

path = 'GVRP_Instances/Table 2'
x = []
y = []
depot = set()
afs = set()
for filename in os.listdir(path):
    with open(os.path.join(path, filename), 'r') as f:
        lines = f.readlines()[2:26]
        depot.add((float(lines[0].split()[2]), float(lines[0].split()[3])))
        for i in range(1, 4):
            afs.add((float(lines[i].split()[2]), float(lines[i].split()[3])))
        for line in lines[6:]:
            line = line.strip()
            if not line.startswith('C'):
                print(f'the format of {filename} is not consistent with the others')
                print(line)
                break
            line = line.split()
            x.append(float(line[2]))
            y.append(float(line[3]))

assert len(depot) == 1 and len(afs) == 3

# min(x): -79.44, max(x): -75.51; min(y): 36.03, max(y): 39.49
print('min(x): %.2f, max(x): %.2f; min(y): %.2f, max(y): %.2f' % (min(x), max(x), min(y), max(y)))
depot = list(depot)
plt.scatter(depot[0][0], depot[0][1], s=200, c='black', marker='*')
afs = list(afs)
afs_x = [cor[0] for cor in afs]
afs_y = [cor[1] for cor in afs]
plt.scatter(afs_x, afs_y)
plt.scatter(x, y)


def cal_dis(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))*4182.44949     # miles
    # c = 2 * atan2(sqrt(a), sqrt(1 - a)) * 4182.44949
    return c


dis = []
afs2depot = [cal_dis(depot[0][0], depot[0][1], afs1, afs2) for afs1, afs2 in zip(afs_x, afs_y)]
for lon, lat in zip(x, y):
    d = cal_dis(depot[0][0], depot[0][1], lon, lat)
    if d > 150:        # at most 150 miles for one way
        afs2cus = [cal_dis(lon, lat, afs1, afs2) for afs1, afs2 in zip(afs_x, afs_y)]
        dis_by_afs = min([d1+d2 for d1, d2 in zip(afs2cus, afs2depot)])
        if dis_by_afs > 300:
            print(f'point ({lon}, {lat}) cannot be satisfied')
            dis.append(d*2)
        else:
            plt.scatter(lon, lat, marker='x', c='r', s=100)
            dis.append(d + dis_by_afs)
    else:
        dis.append(d*2)

plt.show()
