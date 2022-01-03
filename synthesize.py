import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import os 

CURDIR = os.path.dirname(os.path.abspath(__file__))

def read_frames(vid_path): 
    cap = cv2.VideoCapture(vid_path)
    frame_arr = []
    ret = True
    while(ret):
        ret, frame = cap.read()
        if ret: 
            frame_arr.append(frame)
        #np_frame = cv2.imread('video', frame) # does not work
        #np_frame = np.asarray(cv2.GetMat(frame)) # does not work
        #print(np_frame.shape)
    print('reading frames done')
    cap.release()
    return frame_arr

def linear_map(): 
    int_map = np.zeros((256,256),dtype = np.uint8)
    for row in range(256): 
        for col in range(256): 
            sum = row + col
            if sum > 255: 
                int_map[row,col] = 255
            else : 
                int_map[row,col] = sum
            
    # int_map = cv2.applyColorMap(int_map,cv2.COLORMAP_JET)
    return int_map

def nonlinear_map():
    int_map = np.zeros((256,256),dtype = np.uint8)
    for row in range(256): 
        for col in range(256): 
            noise = np.random.rand() * 20
            intensity = np.sqrt((row * row) + (col * col)) + noise 
            if intensity > 230: 
                int_map[row, col] = 255
            else: 
                int_map[row, col] = intensity
    return int_map

# def synthesize_nonlinear(trans, ref): 
#     original_shape = trans.shape
#     original_img = ref
#     trans = np.float64(trans) 
#     ref = np.float64(ref)
#     alpha = np.random.uniform(0.2,1)
#     beta = np.random.uniform(0,5)
#     ref = cv2.GaussianBlur(ref,(5,5),beta)





def synthesize_linear(trans, ref, alpha, beta, int_map): 
    original_shape = trans.shape
    original_img = ref 

    trans = np.float64(trans)
    ref = np.float64(ref)
    ref = cv2.GaussianBlur(ref,(5,5),beta)

    trans = np.reshape(trans,(-1,3))
    ref = np.reshape(ref,(-1,3))
    new_frame = np.full_like(trans,0)
    ref = ref * alpha 

    # int_map is 255 x 255 
    trans_int = np.int64(trans)
    ref_int = np.int64(ref)
    # ref and trans aquired 
    for i in range(len(trans)): 
        # N x 3 
        new_frame[i] = int_map[trans_int[i],ref_int[i]]
    
    new_frame = np.reshape(new_frame,original_shape)


    ref = np.reshape(ref,original_shape)
    trans = np.reshape(trans,original_shape)
    ref = np.uint8(ref)
    trans= np.uint8(trans)
    new_frame = np.uint8(new_frame)

    return new_frame


def synthesize_video(trans_frames,reflec_frames,int_map,alpha,beta): 
    frames = [trans_frames, reflec_frames]
    l = list(map(lambda x: len(x), frames))
    index = np.argmin(l)
    frame_length = l[index]
    new_frame_arr = []
    for idx in range(frame_length): 
        print(idx,' done')
        img = synthesize_linear(trans_frames[idx], reflec_frames[idx],alpha,beta,int_map)
        new_frame_arr.append(img)
    

    return new_frame_arr
    # cv2.imshow('original',original_img)
    # cv2.imshow('ref',ref)
    # cv2.imshow('trans',trans)
    # cv2.imshow('new',new_frame)



def main(): 
    int_map = linear_map() 
    nonlinear_int_map = nonlinear_map()

    trans = read_frames(os.path.join(CURDIR,'1.mp4'))
    reflec = read_frames(os.path.join(CURDIR,'3.mp4'))
    alpha = np.random.uniform(0.2,1)
    beta = np.random.uniform(0,5)

    # linear_vid = synthesize_video(trans, reflec, int_map,alpha,beta) 
    # nonlinear_vid = synthesize_video(trans, reflec, nonlinear_int_map,alpha,beta)

    # for i in nonlinear_vid: 
    #     cv2.imshow('image',i)
    #     cv2.waitKey(100)



    # size = (trans[0].shape[1], trans[1].shape[0])

    # out = cv2.VideoWriter('linear.avi', cv2.VideoWriter_fourcc(*'DIVX'),25,size)
    # for img in linear_vid:
    #     out.write(img)
    # out.release()

    # out1 = cv2.VideoWriter('nonlinear.avi', cv2.VideoWriter_fourcc(*'DIVX'),25,size)
    # for img1 in nonlinear_vid:
    #     out1.write(img1)
    # out1.release()
   

    int_map = cv2.applyColorMap(int_map,cv2.COLORMAP_JET)
    nonlinear_int_map = cv2.applyColorMap(nonlinear_int_map,cv2.COLORMAP_JET)
    cv2.imshow('image',int_map)
    a = cv2.applyColorMap(nonlinear_map(),cv2.COLORMAP_JET)
    cv2.imwrite('img.png',a)
    cv2.imshow('nonlin_image',cv2.applyColorMap(nonlinear_map(),cv2.COLORMAP_JET))
    cv2.imshow('nonlin_image1',nonlinear_map())
    cv2.imshow('nonlin_image2',nonlinear_map())
    cv2.imshow('nonlin_image3',nonlinear_map())
    cv2.waitKey(0)

if __name__ == '__main__': 
    main()