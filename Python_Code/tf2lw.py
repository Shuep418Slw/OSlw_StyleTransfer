import numpy as np

global_type="%f,"

def ConvToStr(c):
    re_str=""
    c_shape=c.shape
    length=c_shape[0]*c_shape[1]
    for n in range(0,c_shape[3]):
        re_str += "/***/\n"
        for m in range(0,c_shape[2]):
            temp=np.reshape(c[:,:,m,n],(1,length))[0]
            for i in temp:
                re_str+=(global_type%i)
            re_str+="\n"
    return re_str[0:len(re_str)-2]


def CnnInToStr(c):
    re_str=""
    c_shape=c.shape
    length=c_shape[1]*c_shape[2]
    print(c_shape[0])
    print(c_shape[3])
    for n in range(0,c_shape[0]):
        for m in range(0,c_shape[3]):
            temp=np.reshape(c[n,:,:,m],(1,length))[0]
            for i in temp:
                re_str+=(global_type%i)
            re_str+="\n"
            print(m)
        print(n)
    return re_str[0:len(re_str)-2]

def CnnInWriteCsv(c,name_csv):
    f=open(name_csv,"w+")
    c_shape=c.shape
    length=c_shape[1]*c_shape[2]
    for n in range(0,c_shape[0]):
        for m in range(0,c_shape[3]):
            temp=np.reshape(c[n,:,:,m],(1,length))[0]
            wr_str=""
            for i in temp:
                wr_str+=(global_type%i)
            wr_str+="\n"
            f.write(wr_str)
            print(m)
        print(n)
    f.close()

def FisrtFullConTrans(a,shape_l):
    buf=np.zeros_like(a)
    length=shape_l[0]*shape_l[1]
    a_shape=a.shape
    for n in range(0,a_shape[0]):
        n1=n%shape_l[2]
        n2=n/shape_l[2]
        m=int(length*n1+n2)
        buf[m,:]=a[n,:]
    #print(buf)
    buf2=np.reshape(buf,(1,a_shape[0]*a_shape[1]))[0]
    restr=""
    for t in buf2:
        restr+=(global_type%(t))

    return restr[0:len(restr)-1]

def bi2str(bi):
    list_data=list(bi)
    ss=str()
    for t in list_data:
        ss+=(global_type%t)
    return ss[0:len(ss)-1]

def we2str(we):
    we=np.reshape(we,(1,np.size(we)))[0]
    list_data=list(we)
    ss=str()
    for t in list_data:
        ss+=(global_type%t)
    return ss[0:len(ss)-1]


if __name__ == '__main__':
    pass
