import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

def showPlot(lenfr,rsarr,average):
    #x - frame
    #y - result
    results = np.array(rsarr)
    indices = np.arange(lenfr)
    plt.figure(figsize=(10, 5))
    plt.plot(indices, results, color='blue', label='Video motion ', linewidth=2)
    plt.axhline(y=average, color='r', linestyle='--', label=f'Average: {average:.2f}')
    plt.ylim(-2, 2)  # Set y-axis limits
    plt.title('Video motion difference')
    plt.xlabel('Frame')
    plt.ylabel('Result')
    plt.grid()
    plt.legend()
    plt.show()

def storeFrames (video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return  
    frames = []
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break 
        frames.append((ret,frame))
        
    #print(f"CALC LEN Frames : {len(frames)}")
    return frames
def avgOfDiff(frames):
    if frames==None or len(frames) == 0 :
        print("Error: Frames are empty!")
        return
    last_mean = 0
    avg = 0
    arr = []
    for f in frames:
        ret,frame = f
        if not ret:
                break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.abs(np.mean(gray) - last_mean)
        avg+=result
        last_mean = np.mean(gray)
        arr.append(result)

    return (arr,float(avg/len(frames)))

def gradient(arrAvg,avg):
    frame = []
    gradientarr = []
    for i in (range(len(arrAvg))):
            pos=i+1
            if (i+1)==len(arrAvg):
                pos=0
            val = (arrAvg[pos]-arrAvg[i-1])/2
            if abs(val)>=avg: 
                gradientarr.append(val)
                frame.append(i)
            else:
                gradientarr.append(0.0)

    return (gradientarr,frame)
def avgFrames(avgRes,avg):
    avgs = []
    for i in range(len(avgRes)):
        if avgRes[i]>=avg:
            avgs.append(i)
    return avgs
def detect_motion_in_video(frames,avgDiff):
    detected  = []
    noDetect  = []
    
    last_mean = 0
    detected_motion = True
    if frames==None or len(frames) == 0 :
        print("Error: Frames are empty!")
        return
    frm = 0
    avg = 0
    for f in frames:
        ret, frame = f

        if not ret:
            break  # Break if no more frames
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = np.abs(np.mean(gray) - last_mean)
        last_mean = np.mean(gray)

        if result < avgDiff :
            detected_motion = False
        # Save a frame if its detected 
        if detected_motion : 
            detected.append(frm)
        
        #noDetect.append(frm)

        frm+=1
    #print(f"LEN Frames : {len(frames)}")
    #print(f"AVG CALC : {avg/len(frames)}")
    return (detected,noDetect)



def showFrames(detected,frames,ttl):

    for d in detected:
        ret,frame = frames[int(d)]
        if not ret:
            break  # Break if no more frames 

        cv2.imshow(ttl+f" {d}", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_unique_frames(frames, threshold=30):
    uinds = []
    if frames==None or len(frames) == 0 :
        print("Error: Frames are empty!")
        return
     

    unique_frames = []
    retf,framef = frames[0]
    prev_frame = framef
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for easier comparison
    unique_frames.append(prev_frame)
    i =0
    for f in frames:
        ret,frame = f
        if not ret:
                break

        # Convert current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the current frame and the previous frame
        diff = cv2.absdiff(prev_frame, gray_frame)

        # Check if the difference is significant (based on the threshold)
        non_zero_count = np.count_nonzero(diff)
        if non_zero_count > threshold:
            unique_frames.append(gray_frame)
            uinds.append(i)
            prev_frame = gray_frame  # Update the previous frame to the current frame

        i+=1
    return (unique_frames,uinds)

def get_unique_frames_from_array(frames, ssim_threshold=0.95):
    unique_frames = []

    # Extract the first frame from the array
    success, prev_frame = frames[0]
    if not success:
        print("Error: First frame is invalid.")
        return []

    # Convert the first frame to grayscale for comparison
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    unique_frames.append((success, prev_frame))  # Store the first frame
    uind = []
    i=0
    # Loop through the remaining frames
    for success, frame in frames[1:]:
        if not success:
            continue

        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the Structural Similarity Index (SSIM) between the previous frame and the current frame
        score, _ = ssim(prev_gray, gray_frame, full=True)

        # If the SSIM score is below the threshold, consider the frame unique
        if score < ssim_threshold:
            uind.append(i)
            unique_frames.append((success, frame))  # Store the unique frame
            prev_gray = gray_frame  # Update the previous frame
        i+=1
    return unique_frames,uind



def get_unique_frames_from_array(frames, ssim_threshold=0.95, resize_factor=0.5):
    unique_frames = []

    # Extract the first frame from the array
    success, prev_frame = frames[0]
    if not success:
        print("Error: First frame is invalid.")
        return []

    # Resize and convert the first frame to grayscale for comparison
    prev_frame_resized = cv2.resize(prev_frame, (0, 0), fx=resize_factor, fy=resize_factor)
    prev_gray = cv2.cvtColor(prev_frame_resized, cv2.COLOR_BGR2GRAY)
    unique_frames.append((success, prev_frame))  # Store the original frame
    uind = []
    i=0
    # Loop through the remaining frames
    for success, frame in frames[1:]:
        if not success:
            continue

        # Resize and convert the current frame to grayscale
        frame_resized = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Compute the Structural Similarity Index (SSIM) between the previous frame and the current frame
        score, _ = ssim(prev_gray, gray_frame, full=True)

        # If the SSIM score is below the threshold, consider the frame unique
        if score < ssim_threshold:
            uind.append(i)
            unique_frames.append((success, frame))  # Store the original frame
            prev_gray = gray_frame  # Update the previous frame
        i+=1
    return unique_frames,uind

def getDuration(st):
    duration = datetime.now() - st
    sec = duration.seconds
    msec = duration.microseconds // 1000
    return sec,msec
#Maka frames
avgDiff = 0.0

st = datetime.now()
frames= storeFrames('../classroom.mp4')
#print(f"FRAMES : {len(frames)}")
print(f"FRAMES TIME :{getDuration(st)}")
st = datetime.now()

uframes,uframind = get_unique_frames(frames)
#print(f"unique frames : {len(uframes)}")

print(f"unique frames time :  {getDuration(st)}")
st = datetime.now()


uff,uind = get_unique_frames_from_array(frames)
#print(f"uff : {len(uff)}")
#print(f"ind : {uind}")

print(f"ssim time : {getDuration(st)}")
st = datetime.now()



uff_sm,uind_sm = get_unique_frames_from_array(frames)
#print(f"uff_sm : {len(uind_sm)}")
#print(f"ind : {uind_sm}")
print(f"ssim time with small : {getDuration(st)}")

#st = datetime.now()

#showFrames(uind,frames,"Detected UFF : ")
showFrames(uind_sm,frames,"Detected UFF_SM : ")


#arrRes,avgDiff = avgOfDiff(frames)
#print(f"FR AVG : {avgDiff}")
#print(frames[0])
#showPlot(len(frames),arrRes,avgDiff)
#gradientarr,intframes = gradient(arrRes,avgDiff)

#showPlot(len(gradientarr),gradientarr,avgDiff)
#showFrames(intframes,frames,"Detected Gradient : ")

#avgintr = avgFrames(arrRes,avgDiff)
#showFrames(avgintr,frames,"Detected AVG : ")



#showPlot(len(gradientarr),gradientarr,avgDiff)
#showFrames(intframes,frames,"Detected Grad")




#inds = input("Give me indexes seporated by space :").split()
#print(inds) 0 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950
#showFrames(inds,frames,"Detected Grad")
#detected_frames,notd = detect_motion_in_video(frames,avgDiff)
#showFrames(detected_frames,frames,"Detected")
#showFrames(notd,frames,"Not Detected")
#print(detected_frames[0])
#showFrames(detected_frames,frames,"Detected")
#showFrames(notd,frames,"Not Detected")