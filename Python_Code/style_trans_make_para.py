#copy *.ckpt to the same dir
#and running this code
#it will generate style_trans_we.h and style_trans_we.py

from tensorflow.python import pywrap_tensorflow
import numpy as np
import tf2lw

tf2lw.global_type="%.10f,"

fid=open("style_trans_we.h","w+")
fnp=open("style_trans_we.py","w+")



fid.write("#ifndef ST_WE_H_\n#define ST_WE_H_\n")
fnp.write("import numpy as np\n")


#***********************

#choose different model
#checkpoint_path = "star.ckpt-34000"
#checkpoint_path = "wave.ckpt-25000"
#checkpoint_path = "sumiao.ckpt-35000"
#checkpoint_path = "shuimo.ckpt-26000"
checkpoint_path = "bitflow.ckpt-25000"

#************************
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    if (key.find("Adam") != -1):
        continue
    namestr=key.replace("/","_")
    print("tensor_name: ", namestr)

    data_np=reader.get_tensor(key)
    data_shape=np.shape(data_np)
    print(data_shape)

    data_str=""
    data_str2=str(list(np.reshape(data_np,np.size(data_np))))

    if(key.find("IN_beta")!=-1):
        data_np=np.reshape(data_np,(1,np.size(data_np)))
        data_str=tf2lw.we2str(data_np)
    elif(key.find("IN_gamma")!=-1):
        data_np = np.reshape(data_np, (1, np.size(data_np)))
        data_str = tf2lw.we2str(data_np)
    elif(key.find("weight")!=-1):
        data_str=tf2lw.ConvToStr(data_np)
    else:
        print("err\n")
        continue
    fid.write("const ParaType %s [%d][%d][%d][%d]={%s};\n" % (namestr, data_shape[0], data_shape[1], data_shape[2], data_shape[3],data_str))
    fnp.write("%s = np.reshape(np.array(%s,np.float32),(%d,%d,%d,%d))\n"%(namestr,data_str2,data_shape[0], data_shape[1], data_shape[2], data_shape[3]))


fid.write("#endif\n")

fid.close()
fnp.close()


