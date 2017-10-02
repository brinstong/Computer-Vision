import numpy as np
import cv2
import statistics
import sys
import matplotlib.pyplot as plt
import matplotlib
from sklearn import mixture
import os
import datetime

matplotlib.use("TkAgg")
from collections import deque


class BackgroundSubstractionKNN:
    def __init__(self, k_size):
        self.__kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size,k_size));
        self.__fgbg = cv2.createBackgroundSubtractorKNN();

    def update_state(self, frame):
        self.__fgmask = self.__fgbg.apply(frame)
        self.__fgmask = cv2.morphologyEx(self.__fgmask, cv2.MORPH_OPEN, self.__kernel)
        ret, self.__fgmask_noshadow = cv2.threshold(self.__fgmask, 254, 255, cv2.THRESH_BINARY)
        return self.__fgmask_noshadow;

class BackgroundSubstraction:
    def __init__(self):
        self.__firstframe = None;

    def update_state(self, gray):
        if self.__firstframe is None :
            self.__firstframe = gray

    def subtract_background(self, gray):
        # substraction
        output_image = cv2.absdiff(self.__firstframe,gray)
        output_image = cv2.threshold(output_image,20,255,cv2.THRESH_BINARY)[1]
        # filter
        output_image = cv2.medianBlur(output_image,5)
        # dilate operation: fill the empty hole
        output_image = cv2.dilate(output_image, None, iterations=1)
        #output_image = cv2.erode(output_image, None, iterations=1)
        return output_image

class MHI:
    def __init__(self, MHI_DURATION):
        self.__MHI_DURATION = MHI_DURATION;
        self.__frame_count = 0;
        self.__mhi_count = 0;
        self.__frame_pool = np.empty(self.__MHI_DURATION, dtype=object);

    def MHI_generation(self, start_index):
        start_index = (self.__mhi_count-1)%self.__MHI_DURATION;
        TAO = self.__MHI_DURATION - 1;
        self.__motion_history = np.zeros((self.__frame_pool[0].shape[0], self.__frame_pool[0].shape[1]), np.uint8)
        for i in range(len(self.__frame_pool)):
            img = self.__frame_pool[(i+start_index)%self.__MHI_DURATION]
            for h in range(img.shape[0]):
                for w in range(img.shape[1]):
                    if img[h, w] != 0:
                        self.__motion_history[h, w] = TAO
                    else:
                        self.__motion_history[h, w] = (self.__motion_history[h, w] - 1 if self.__motion_history[h, w] - 1 > 0 else 0)

    def update_state(self, stride, output_image):
        # MHI ----- put new frames into MHI Pool
        if self.__frame_count%stride == 0:
            # form a MHI pool within the most recent "MHI_DURATION" images
            self.__frame_pool[self.__mhi_count%self.__MHI_DURATION] = output_image.copy();
            self.__mhi_count = self.__mhi_count+1;
        self.__frame_count = self.__frame_count + 1;

    @classmethod
    def image4visualize(cls, src):
        motion_history_4visualize = np.zeros((src.shape[0], src.shape[1]), src.dtype)
        cv2.normalize(src, motion_history_4visualize, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("MHI", motion_history_4visualize)

    def generate_show_mhi(self):
        # MHI ------ generate MHI
        if self.__mhi_count >= self.__MHI_DURATION and self.__mhi_count%int(self.__MHI_DURATION/2) == 0:
            self.MHI_generation((self.__mhi_count-1)%self.__MHI_DURATION);
            self.image4visualize(self.__motion_history)
    
    def get_motion_hitory(self):
        if self.__mhi_count >= self.__MHI_DURATION:
            return self.__motion_history
        else:
            return None

class OPTICAL_FLOW:
    def __init__(self, feature_params, lk_params, OFH_duration, med_filter_size):
        self.__feature_params = feature_params;
        self.__lk_params = lk_params;
        self.__frame_num = 0;
        self.__test_speeds = [];
        self.__test_med_speeds = [];
        self.__OFH_DURATION = OFH_duration;
        self.__highest_pool = np.empty(self.__OFH_DURATION, dtype=object);
        self.__speeds = [];
        self.__med_speeds = [];
        self.__startindex = 0;
        self.__med_filter_size = med_filter_size;
        self.__in_moving = None;
    def get_frame_num(self):
        return self.__frame_num;
    def update_med_speeds(self):
        assert(self.__med_filter_size <= self.__OFH_DURATION);
        new_speed = 0;
        if self.__frame_num < self.__med_filter_size:
            new_speed = self.__speeds[-1];
            self.__med_speeds.append(new_speed);
        else:
            sample_speeds = self.__speeds[-self.__med_filter_size:];
            new_speed = statistics.median(sample_speeds);
            self.__med_speeds.append(new_speed);
        if len(self.__med_speeds) > self.__OFH_DURATION:
            self.__med_speeds.pop(0)
        return new_speed;

    def update_state(self, gray):
        max_dist = 0;
        if self.__frame_num == 0:
            self.__p0 = cv2.goodFeaturesToTrack(gray, mask = None, **self.__feature_params);
            self.__highest_pool[(self.__frame_num)%self.__OFH_DURATION] = (0, 0, 0, 0, 0, 0);
            max_dist = 0;
        else:
            if self.__frame_num > 10 and self.__old_max_dist < 0.5:
                self.__p0 = cv2.goodFeaturesToTrack(gray, mask = None, **self.__feature_params);
            self.__p1, st, err = cv2.calcOpticalFlowPyrLK(self.__old_gray, gray, self.__p0, None, **self.__lk_params);
            # Select good points
            if self.__p1 is not None:
                good_new = self.__p1[st==1]
                good_old = self.__p0[st==1]
                max_dist = 0;
                a_max=0; b_max=0; c_max=0; d_max=0;
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    dist = np.math.hypot((a-c),(b-d))
                    if dist > max_dist:
                        max_dist = dist;
                        a_max=a;
                        b_max=b;
                        c_max=c;
                        d_max=d;
                self.__highest_pool[(self.__frame_num)%self.__OFH_DURATION] = (a_max, b_max, c_max, d_max, max_dist, self.__old_max_dist);
                self.__p0 = good_new.reshape(-1,1,2);
        # update old
        self.__old_max_dist = max_dist;
        self.__old_gray = gray.copy();   
        self.__startindex = self.__frame_num;
        self.__speeds.append(max_dist);
        if len(self.__speeds) > self.__OFH_DURATION:
            self.__speeds.pop(0)
        # after speeds is updated
        new_med_speed = self.update_med_speeds();
        self.__test_med_speeds.append(new_med_speed);
        self.__test_speeds.append(max_dist);
        self.__frame_num = self.__frame_num+1;

    def show_history(self, history):
        old_speed = history[0];
        for i,speed in enumerate(history):
            plt.plot([i,i+1],[old_speed, speed],'r')
            old_speed = speed;
        plt.show()
    def check_constant_change(self, data):
        length = len(data);
        increase_count = 0.0;
        for i in range(length-1):
            if data[i+1] > data[i]:
                increase_count = increase_count+1;
        if increase_count/length > 0.75:
            return True
        if (length-increase_count)/length > 0.75:
            return True
    def fall_speed_analysis(self):
        # The med_speed is stored in variable __med_speeds 
        if self.__frame_num >= self.__OFH_DURATION:
            med_speeds = self.__med_speeds;
            max_index = np.argmax(med_speeds);
            highest_speed = med_speeds[max_index];
            lowest_speed_2ndhalf = np.min(med_speeds[max_index:]);
            balanced_graph_check = np.abs(max_index - self.__OFH_DURATION/2) < self.__OFH_DURATION/3*2;
            highest_speed_last = np.max(med_speeds[int(self.__OFH_DURATION/4):]);
            # cond1: speed high enough; 3.5 
            # con2: 2nd half is low enough; 0.5
            # cond3: the highes point is located in the middle; 1/2
            if highest_speed_last < 0.4:
                self.__in_moving = False;
            else:
                self.__in_moving = True;
            #if highest_speed > 3.2:
            #    print(lowest_speed_2ndhalf, balanced_graph_check)
            if highest_speed > 3.2 and lowest_speed_2ndhalf < 0.48 and balanced_graph_check:
                if self.check_constant_change(med_speeds[:max_index+1]):
                    if self.check_constant_change(med_speeds[max_index:]):
                        #print(med_speeds[:max_index+1])
                        #print(med_speeds[max_index:])
                        #exit()
                        return True, self.__in_moving, 1
                    else:
                        return True, self.__in_moving, 0.7
                else:
                    return True, self.__in_moving, 0.5

            if highest_speed < 1.5:
                self.__in_moving = False;
            else:
                self.__in_moving = True;
            return False, self.__in_moving, 0;
        else:
            return False, None, 0;

    def OFH_speeds(self):
        return self.__test_speeds;

    def OFH_med_speeds(self):
        return self.__test_med_speeds;


class BND_BOX:
    def __init__(self, BBH_DURATION, med_filter_size):
        self.__check_fall = False;
        self.__angles = []; # keep the record of the last # of self.__bbh_duration frames
        self.__med_filter_size =  med_filter_size;
        self.__med_angles = [];
        self.__test_med_angles = [];
        self.__bbh_duration = BBH_DURATION;
        self.__frame_num = 0;
        self.__old_angle = 0;
        self.__test_angles = []; # store all the angle history
        self.__bin_size = 18;
    def get_frame_num(self):
        return self.__frame_num;
    def update_med_angles(self):
        assert(self.__med_filter_size <= self.__bbh_duration);
        new_angle = 0;
        if self.__frame_num < self.__med_filter_size:
            new_angle = self.__angles[-1];
            self.__med_angles.append(new_angle);
        else:
            sample_angles = self.__angles[-self.__med_filter_size:];
            new_angle = statistics.median(sample_angles);
            self.__med_angles.append(new_angle);
        if len(self.__med_angles) > self.__bbh_duration:
            self.__med_angles.pop(0)
        return new_angle;

    def pre_processing(self, src):
        kernel1 = np.ones((30,30),np.uint8)
        self.__fgmask = cv2.medianBlur(src, 15)
        self.__fgmask = cv2.dilate(self.__fgmask, kernel1)

    def angle_between(self, p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    def compute_angle(self, vy, vx):
        angle = np.arctan2(vy, vx);
        if angle < 0:
            angle = angle + np.pi;
        angle = angle * 180 / np.pi;
        return angle;

    def update_state(self, img):
        self.pre_processing(img);
        None1, contours, None2 = cv2.findContours(self.__fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        if len(contours) != 0:
            contour = None;
            if len(contours)>1:
                max_area = 0;
                max_label = 0;
                for i in range(len(contours)):
                    contour_area = cv2.contourArea(contours[i]);
                    if contour_area>max_area:
                        max_area = contour_area;
                        max_label = i;
                contour = contours[max_label];
            else:
                contour = contours[0];
            
            #rect = cv2.minAreaRect(contour)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            #if len(contour)>=5:
            #    ellipse = cv2.fitEllipse(contour)
            h, w = img.shape[:2];
            rows, cols = img.shape[:2];
            [vx, vy, cx, cy] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            cv2.line(img, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (255, 255, 255))
            #cv2.line(img, (cols - 1, righty), (0, lefty), (255, 255, 255), 2)
            # cv2.imshow('img', img);
            # cv2.waitKey(10);
            #if self.__frame_num > 700:
            #    exit();
            angle = self.compute_angle(vy, vx);
        else:
            angle = self.__old_angle;
        # update 
        self.__test_angles.append(angle);
        self.__angles.append(angle)
        self.__old_angle = angle;
        if len(self.__angles) > self.__bbh_duration:
            self.__angles.pop(0)
        # after angles is stored, update med_angles
        new_med_angle = self.update_med_angles();
        self.__test_med_angles.append(new_med_angle);
        self.__frame_num = self.__frame_num+1;

    def show_history(self, history):
        old_angle = 0;
        for i,angle in enumerate(history):
            plt.plot([i,i+1],[old_angle, angle],'r')
            old_angle = angle;
        plt.show()
    def twist_range(self, angles, bin_size, shift):
        twisted_angles = [];
        for angle in angles:
            if angle <= bin_size:
                angle = 180 + angle;
            angle = angle + shift;
            twisted_angles.append(angle);
        return twisted_angles
    def argmax2(self, data):
        arr = data.copy();
        largest_index = np.argmax(arr);
        arr = np.delete(arr, largest_index)
        largest_2nd_Index = np.argmax(arr);
        if largest_index <= largest_2nd_Index:
            largest_2nd_Index = largest_2nd_Index+1;
        return largest_index, largest_2nd_Index
    def check_histogram(self, data, bin_size, highest, lowest):
        bins= [];
        for i in range(int(180/bin_size)+3):
            bins.append(bin_size*i)
        hist, bin_edges = np.histogram(data, bins)
        largest_index, largest_2nd_Index = self.argmax2(hist);
        Flag = 0;
        for i,bin_edge in enumerate(bin_edges):
            if lowest < bin_edge:
                low_index2check = i-1;
                Flag = Flag+1;
                lowest = 360;
            if highest < bin_edge:
                large_index2check = i-1;
                Flag= Flag +1;
                highest = 360;
            if Flag == 2:
                break;
        if low_index2check == largest_index or low_index2check == largest_2nd_Index:
            if large_index2check == largest_index or large_index2check == largest_2nd_Index:
                return True;
        return False;
        
    def fall_angle_analysis(self):
        # The med_speed is stored in variable __med_speeds 
        if self.__frame_num >= self.__bbh_duration:
            self.__med_angles;
            highest_index = np.argmax(self.__med_angles);
            lowest_index = np.argmin(self.__med_angles);
            highest = self.__med_angles[highest_index];
            lowest = self.__med_angles[lowest_index]
            if np.abs(highest-lowest)>45:
                bin_size = self.__bin_size;
                if not self.check_histogram(self.__med_angles, bin_size, highest, lowest):
                    shift = np.random.randint(1, bin_size);
                    twisted_angles = self.twist_range(self.__med_angles, bin_size, shift);
                    output = self.twist_range([highest, lowest], bin_size, shift)
                    if not self.check_histogram(twisted_angles, bin_size, output[0], output[1]):
                        shift = np.random.randint(1, bin_size);
                        twisted_angles = self.twist_range(self.__med_angles, bin_size, shift);
                        output = self.twist_range([highest, lowest], bin_size, shift)
                        if not self.check_histogram(twisted_angles, bin_size, output[0], output[1]):
                            return False, 0;
                            shift = np.random.randint(1, bin_size);
                            twisted_angles = self.twist_range(self.__med_angles, bin_size, shift);
                            output = self.twist_range([highest, lowest], bin_size, shift)
                            if not self.check_histogram(twisted_angles, bin_size, output[0], output[1]):
                                return False, 0;
                if np.abs(highest-lowest)>60:
                    return True, 1;
                return True, 0.75;
            return False, 0;
        else:
            return False, 0;

    def BBH_med_angles(self):
        return self.__test_med_angles;
    def BBH_angles(self):
        return self.__test_angles;
        
class FALL_DETECTION:
    def __init__(self, FPS):
        self.__check_fall = False;
        self.__frame_num = 0;
        self.__in_moving = None;
        self.__in_falling = False;
        self.__fall_confirmed = False;
        self.__FPS = int(FPS);
        self.__TIME_TOTAL = int(FPS*2);
        self.__time_left = self.__TIME_TOTAL;
        self.__first_num = None;
        self.__last_num = None;
    def falling_frame_range(self):
        if self.__first_num is None or self.__fall_confirmed == False:
            self.__first_num = -1;
            self.__last_num = -1;
        shift = int(self.__FPS*1.5)
        return self.__first_num-shift, self.__last_num-shift
    def check_frame_num(self):
        return self.__frame_num;
    def update_in_falling(self):
        if self.__time_left == 0:
            self.__in_falling = False;
            self.__time_left = self.__TIME_TOTAL;
            self.__first_num = None;
        self.__time_left = self.__time_left - 1;
        if not self.__in_moving:
            self.__fall_confirmed = True;
            self.__check_fall = True;
            self.__in_falling = False;
            self.__time_left = self.__TIME_TOTAL;
            if self.__last_num is None and self.__check_fall == True:
                self.__last_num = self.__frame_num;
    def update_state(self, decision_speed, in_moving_flag, prob_speed, decision_angle, prob_angle):
        self.__in_moving = in_moving_flag;
        
        if self.__check_fall == True:
            if self.__in_moving == True:
                self.__check_fall = False;
                self.__in_falling = False;
                self.__time_left = self.__TIME_TOTAL;
                # self.__first_num = None;
                # self.__last_num = None;
        else:
            if self.__in_falling == False:
                if decision_speed == False:
                    self.__check_fall = False;
                else:
                    if prob_speed > 1:
                        self.__in_falling = True;
                    else:
                        if decision_angle == True:
                            self.__in_falling = True;
            else:
                if self.__first_num is None and self.__in_falling == True:
                    self.__first_num = self.__frame_num;
                self.update_in_falling();
        self.__frame_num = self.__frame_num+1;
    def check_in_falling(self):
        return self.__in_falling;
    def check_fall(self):
        return self.__check_fall;

def show_history(title, history1, history2):
    fig = plt.figure();
    fig.suptitle(title);
    old_temp = history1[0];
    for i,temp in enumerate(history1):
        med_plot, = plt.plot([i,i+1],[old_temp, temp],'r',linewidth=1.0, label = 'median blurred')
        old_temp = temp;
    old_temp = history2[0];
    for i,temp in enumerate(history2):
        ori_plot, = plt.plot([i,i+1],[old_temp, temp],'--',color='lime', linewidth=1, label = 'raw data ')
        old_temp = temp;
    plt.legend(handles=[ori_plot,med_plot])
    plt.show()

def show_med_filter(optical_instance, bnd_box_instance):
    med_speeds = optical_instance.OFH_med_speeds()
    speeds = optical_instance.OFH_speeds();
    show_history('speed history', med_speeds, speeds)
    med_angles = bnd_box_instance.BBH_med_angles()
    angles = bnd_box_instance.BBH_angles();
    #show_history('angle history', med_angles, angles)


def process_video(img_name, Output_path, Flag_video_generate, Flag_video_show):
    # Open video
    cap = cv2.VideoCapture(img_name)
    # CONSTANT
    FPS = cap.get(5);#25
    IPS = 5;
    stride = int(FPS/IPS);
    # Parameter for Flag_video_generate
    if Flag_video_generate:
        fps = cap.get(5);
        # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        size = (640, 480);
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
        vout = cv2.VideoWriter()
        if img_name == 0:
            output_filename = '_camera_output.mp4';
        else:
            output_filename = os.path.splitext(os.path.basename(img_name))[0]+'.mp4'
        Output_fullname = Output_path+output_filename;
        if os.path.exists(Output_fullname):
            os.remove(Output_fullname)
        if img_name == 0:
            success = vout.open(Output_fullname, fourcc, int(20), size, True)
        else:
            success = vout.open(Output_fullname, fourcc, fps, size, True)
    if Flag_video_show:
        a =1;

    # MHI  ----- MHI parameter setting
    # length of history in record
    # MHI_DURATION = 6;
    # mhi_instance = MHI(MHI_DURATION);

    # Background Subtraction Parameters
    # back_sub_instance = BackgroundSubstraction();
    K_SIZE = 3;
    back_subknn_instance = BackgroundSubstractionKNN(K_SIZE);

    # OPTICAL FLOW Parameters
    # params for ShiTomasi corner detection
    # coffee 150, 0.15, 5, 7 (40,40) , 3, 
    feature_params = dict( maxCorners = 150,
                           qualityLevel = 0.15,
                           minDistance = 5,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (40,40),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    OFH_DURATION = int(1.5*FPS)+1;
    med_filter_size = int(FPS/2);
    optical_instance = OPTICAL_FLOW(feature_params, lk_params, OFH_DURATION, med_filter_size);

    # Parameters for BND_BOX
    BBH_DURATION = int(1.5*FPS)+1;
    med_filter_size = BBH_DURATION;
    bnd_box_instance = BND_BOX(BBH_DURATION, med_filter_size);

    # Parameters for Fall_detection:
    fall_detec_instance = FALL_DETECTION(FPS);
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            # show_med_filter(optical_instance, bnd_box_instance)
            # print(fall_detec_instance.falling_frame_range());
            return fall_detec_instance.falling_frame_range()
            break
        frame_ori = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_CUBIC);
        frame = cv2.resize(frame, (320, 240), interpolation = cv2.INTER_CUBIC);
        # break at the end of file
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # 1st: ---- update state
        # BS ------ START
        # initialization
        # back_sub_instance.update_state(gray);
        # backgroudn_subtraction
        # output_image = back_sub_instance.subtract_background(gray);
        # BS --KNN 
        output_image = back_subknn_instance.update_state(frame);
        # BS ------ END

        # MHI ----- START -----------------
        # MHI ----- put new frames into MHI Pool
        # stride = int(FPS/IPS);
        # mhi_instance.update_state(stride, output_image);
        # MHI ------ generate MHI
        # mhi_instance.generate_show_mhi();
        # motion_history = mhi_instance.get_motion_hitory();
        # MHI ------- END ------------------

        # OPTICAL FLOW ---------- START ---------------
        optical_instance.update_state(gray);
        (decision_speed, in_moving_flag, prob_speed) = optical_instance.fall_speed_analysis();
        #if decision_speed == True:
        #    print(optical_instance.get_frame_num())
        # OPTICAL FLOW ---------- END -----------------
        ### BND BOX --------  Start  --------------
        bnd_box_instance.update_state(output_image);
        (decision_angle, prob_angle) = bnd_box_instance.fall_angle_analysis();
        # print(decision_angle)
        # if decision_angle == True:
        #    print(bnd_box_instance.get_frame_num())
        ###  BND BOX --------  End   ---------------
    # 2nd: ---- make a decision about falling
        fall_detec_instance.update_state(decision_speed, in_moving_flag, prob_speed, decision_angle, prob_angle);
        if Flag_video_generate or Flag_video_show:
            frame2show = frame_ori.copy();
        if fall_detec_instance.check_in_falling() == True:
            if Flag_video_generate or Flag_video_show:
                frame2show = frame_ori.copy();
                cv2.putText(frame2show, "Possible Fall...", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=1)
            #print(fall_detec_instance.check_frame_num())
            #print("Possible Falling")

        if fall_detec_instance.check_fall():
            if Flag_video_generate or Flag_video_show:
                frame2show = frame_ori.copy();
                cv2.putText(frame2show, "Fall Detected!!!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), thickness=1)
            #print(fall_detec_instance.check_frame_num())
            #print("Fall detected")
        if Flag_video_generate == True:
            vout.write(frame2show)
        if Flag_video_show == True:
            k =1;
            cv2.imshow("FallDetection", frame2show)
            k = cv2.waitKey(1)
            if k == 27:
                break
        #cv2.imshow("frame", output_image)
        #if cv2.waitKey(1) & 0xFF == ord('q') :
        #   break
        #cv2.imshow("orig",frame)
        #  user termination option
    cap.release()
    #cv2.destroyAllWindows()

def process_one_video(dataset_name, i):
    img_path_prefix = 'dataset/'+dataset_name+'/Videos/video ('
    img_path_postfix = ').mov'
    Output_path = 'Output/_'
    img_full_name = img_path_prefix+str(i)+img_path_postfix;
    fall_range = process_video(img_full_name, Output_path, False, False);
    print(fall_range)
def intersection_over_union(a, b):
    u_left = min(a[0], b[0])
    u_right = max(a[1], b[1])
    i_left = max(a[0], b[0])
    i_right = min(a[1], b[1])
    return (i_right-i_left)/(u_right-u_left)

def process_dataset(dataset_name, size):
    img_path_prefix = 'dataset/'+dataset_name+'/Videos/video ('
    img_path_postfix = ').mov'
    Output_path = 'Output/_'
    annotion_path_prefix ='dataset/'+dataset_name+'/Annotation_files/video ('
    annotion_path_postfix = ').txt'

    sum = 0;
    for i in range(1,size+1):
        img_full_name = img_path_prefix+str(i)+img_path_postfix;
        fall_range = process_video(img_full_name, Output_path, False, False);
        first_truth = None;
        last_truth = None;
        with open(annotion_path_prefix+str(i)+annotion_path_postfix) as f:
            line1 = f.readline()
            line2 = f.readline()
            first_truth = int(line1)
            last_truth = int(line2)
        iou = intersection_over_union(fall_range, (first_truth, last_truth))
        print(i, iou , fall_range, (first_truth, last_truth))
        if iou > 0:
            sum = sum + iou;
    print(sum/size)

def process_camera():
    Output_path = 'Output/_'
    fall_range = process_video(0, Output_path, False, True);

if __name__ == '__main__':
    np.random.seed(2)
    # process_dataset('Coffee_room_01', 48)
    # process_dataset('Home_01', 30)
    # rocess_one_video('Coffee_room_01', 48)
    process_camera()


