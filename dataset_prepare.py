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

def synthesize_nonlinear(trans, ref): 
    original_shape = trans.shape
    original_img = ref
    trans = np.float64(trans) 
    ref = np.float64(ref)
    alpha = np.random.uniform(0.2,1)
    beta = np.random.uniform(0,5)
    ref = cv2.GaussianBlur(ref,(5,5),beta)





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

    return new_frame, ref


def synthesize_video(trans_frames,reflec_frames,int_map,alpha,beta): 
    frames = [trans_frames, reflec_frames]
    l = list(map(lambda x: len(x), frames))
    index = np.argmin(l)
    frame_length = l[index]
    new_frame_arr = []
    new_ref_arr = []
    for idx in range(frame_length): 
        print(idx,' done')
        img, imgref = synthesize_linear(trans_frames[idx], reflec_frames[idx],alpha,beta,int_map)
        new_frame_arr.append(img)
        new_ref_arr.append(imgref)

    return new_frame_arr, new_ref_arr

def synthesize_video1(trans_frames, reflec_frames, int_map, alpha, beta): 
    # frames = [trans_frames, reflec_frames]
    # l = list(map(lambda x: len(x), frames))
    # index = np.argmax(l)
    # frame_length = l[index]
    frame_length = len(trans_frames)
    new_frame_arr = []
    new_ref_arr = []
    for idx in range(frame_length): 
        print(idx,' done')
        img, imgref = synthesize_linear(trans_frames[idx], reflec_frames[0],alpha,beta,int_map)
        new_frame_arr.append(img)
        new_ref_arr.append(imgref)

    return new_frame_arr, new_ref_arr




def main(): 
    int_map = linear_map() 
    nonlinear_int_map = nonlinear_map()
    video_path = os.path.join(CURDIR,'480p')
    folder_lists = os.listdir(video_path)
    trans_folders = folder_lists[0:int(len(folder_lists)/2)]
    reflect_folders = folder_lists[int(len(folder_lists)/2):]
    
    trans_filenames = []    
    reflec_filenames = []    
    for t in trans_folders: 
        fold = os.path.join(video_path,t)
        files = os.listdir(fold)
        trans_filenames.append(files)

    for r in reflect_folders: 
        fold = os.path.join(video_path,r)
        files = os.listdir(fold)
        reflec_filenames.append(files)

    for i in range(len(trans_filenames)): 
        tlen = len(trans_filenames[i])
        rlen = len(reflec_filenames[i])
        shorter = min([tlen, rlen]) 
        

        
        alpha = np.random.uniform(0.2,0.9)
        beta = np.random.uniform(0,5)
        timages = []
        rimages = []
        for j in range(shorter): 
            tfile = os.path.join(video_path,trans_folders[i],trans_filenames[i][j])
            rfile = os.path.join(video_path,reflect_folders[i],reflec_filenames[i][j])
            if "@" in tfile: 
                continue
            timg = cv2.imread(tfile)
            rimg = cv2.imread(rfile)
            timg = cv2.resize(timg,(854,480))
            rimg = cv2.resize(rimg,(854,480))
            timages.append(timg)
            rimages.append(rimg)
        
        # for ind, image in enumerate(timages): 
        #     temp_transdir = os.path.join(CURDIR,'dataset1','trans',str(i))
        #     if not os.path.isdir(temp_transdir):
        #         os.mkdir(os.path.join(temp_transdir))
        #     save_transfile = os.path.join(temp_transdir,str(ind)+'.jpg') 
        #     cv2.imwrite(save_transfile,timages[ind])
        

        # synth_images, ref_images = synthesize_video(timages, rimages, nonlinear_int_map,alpha,beta)
        synth_images, ref_images = synthesize_video1(timages, rimages, nonlinear_int_map,alpha,beta)
        
        with open(os.path.join(CURDIR,'dataset2','intmap.npy'), 'wb') as f: 
            np.save(f,int_map)

        synthetic_dir = os.path.join(CURDIR,'dataset2','synthetic',str(i))
        if not os.path.isdir(synthetic_dir):
            os.makedirs(synthetic_dir)

        temp_refdir = os.path.join(CURDIR,'dataset2','reflect',str(i))
        if not os.path.isdir(temp_refdir):
            os.makedirs(temp_refdir)
        
        temp_transdir = os.path.join(CURDIR,'dataset2','trans',str(i))
        if not os.path.isdir(temp_transdir):
            os.makedirs(os.path.join(temp_transdir))


        for ind, image in enumerate(synth_images):
            save_file = os.path.join(synthetic_dir,str(ind)) 
            save_file = save_file + '.jpg'
            cv2.imwrite(save_file,image)

            save_reffile = os.path.join(temp_refdir,str(ind)+'.jpg') 
            # cv2.imwrite(save_reffile,rimages[ind])
            cv2.imwrite(save_reffile,ref_images[ind])

            save_transfile = os.path.join(temp_transdir,str(ind)+'.jpg') 
            cv2.imwrite(save_transfile,timages[ind])



if __name__ == '__main__': 
    main()