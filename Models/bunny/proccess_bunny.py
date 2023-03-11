import os

objFilePath = './bunny_low_resolution.obj'

points = []

with open(objFilePath) as file:
    while 1:
        line = file.readline()
        if not line:
            print("not a line")
            break
        points.append(line)
        if line=='':
            break
        strs = line.split(" ")

filename = 'bunny_x.obj'
with open(filename, 'w') as file_object:
    for i in range(0,len(points)):
        proc_str = points[i]
        strs = points[i].split(" ")
        if(strs[0]=="v"):
            strs[1] = str(float(strs[1])+1)
            strs[2] = str(float(strs[2]))
            strs[3] = str(float(strs[3]))
            proc_str = strs[0]+" "+strs[1]+" "+strs[2]+" "+strs[3]
        if(strs[0]=="vt"):
            file_object.write("\n")
        # print(proc_str)
        file_object.write(proc_str)




points = []

with open(objFilePath) as file:
    while 1:
        line = file.readline()
        if not line:
            print("not a line")
            break
        points.append(line)
        if line=='':
            break
        strs = line.split(" ")

filename = 'bunny_z.obj'
with open(filename, 'w') as file_object:
    for i in range(0,len(points)):
        proc_str = points[i]
        strs = points[i].split(" ")
        if(strs[0]=="v"):
            strs[1] = str(float(strs[1]))
            strs[2] = str(float(strs[2]))
            strs[3] = str(float(strs[3])+1)
            proc_str = strs[0]+" "+strs[1]+" "+strs[2]+" "+strs[3]
        if(strs[0]=="vt"):
            file_object.write("\n")
        # print(proc_str)
        file_object.write(proc_str)
