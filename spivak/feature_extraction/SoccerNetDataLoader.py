# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.
#
# This file incorporates work covered by the following copyright and permission
# notice:
#   Copyright (c) 2018 Silvio Giancola
#   Licensed under the terms of the MIT license.
#   You may obtain a copy of the MIT License at https://opensource.org/licenses/MIT

# This file contains pieces of code taken from the SoccerNet/DataLoader.py
# file of the SoccerNet pip package, version 0.1.9. You can find the package
# at the following link.
# https://pypi.org/project/SoccerNet/0.1.9/
# At Yahoo Inc., the code was modified and new code was added, so that FrameCV
# could handle vertical videos when using the "crop" transform.

import logging
from math import floor, ceil

import cv2
import imutils
import moviepy.editor
import numpy as np
import skvideo.io
from tqdm import tqdm


class FrameCV:

    def __init__(self, video_path, FPS=2, transform=None, start=None, duration=None):
        """Create a list of frame from a video using OpenCV.

        Keyword arguments:
        video_path -- the path of the video
        FPS -- the desired FPS for the frames (default:2)
        transform -- the desired transformation for the frames (default:2)
        start -- the desired starting time for the list of frames (default:None)
        duration -- the desired duration time for the list of frames (default:None)
        """

        self.FPS = FPS
        self.transform = transform
        self.start = start
        self.duration = duration

        # read video
        vidcap = cv2.VideoCapture(video_path)
        # read FPS
        self.fps_video = vidcap.get(cv2.CAP_PROP_FPS)
        # read duration
        self.time_second = _get_duration(video_path)

        # loop until the number of frame is consistent with the expected number of frame, 
        # given the duratio nand the FPS
        good_number_of_frames = False
        while not good_number_of_frames: 

            # read video
            vidcap = cv2.VideoCapture(video_path)
            
            # get number of frames
            self.numframe = int(self.time_second*self.fps_video)
            
            # frame drop ratio
            drop_extra_frames = self.fps_video/self.FPS

            # init list of frames
            self.frames = []

            # TQDM progress bar
            pbar = tqdm(range(self.numframe), desc='Grabbing Video Frames', unit='frame')
            i_frame = 0
            ret, frame = vidcap.read()

            # loop until no frame anymore
            while ret:
                # update TQDM
                pbar.update(1)
                i_frame += 1
                
                # skip until starting time
                if self.start is not None:
                    if i_frame < self.fps_video * self.start:
                        ret, frame = vidcap.read()
                        continue

                # skip after duration time
                if self.duration is not None:
                    if i_frame > self.fps_video * (self.start + self.duration):
                        ret, frame = vidcap.read()
                        continue
                        

                if (i_frame % drop_extra_frames < 1):

                    if self.transform == "resize256crop224":
                        # crop keep the central square of the frame
                        frame = imutils.resize(frame, height=256)  # keep aspect ratio
                        # number of pixel to remove per side
                        off_h = int((frame.shape[0] - 224)/2)
                        off_w = int((frame.shape[1] - 224)/2)
                        frame = frame[off_h:-off_h,
                                      off_w:-off_w, :]  # remove pixel at each side

                    elif self.transform == "crop":
                        # crop to remove the sides or top and bottom of the
                        # frame.
                        height = frame.shape[0]
                        width = frame.shape[1]
                        if width >= height:
                            # resize while keeping the aspect ratio
                            frame = imutils.resize(frame, height=224)
                            # number of pixel to remove per side
                            new_width = frame.shape[1]
                            off_side = int((new_width - 224)/2)
                            # remove them
                            frame = frame[:, off_side:-off_side, :]
                        else:
                            frame = imutils.resize(frame, width=224)
                            # number of pixel to remove from top and bottom
                            new_height = frame.shape[0]
                            half_excess = (new_height - 224) / 2.0
                            off_top = floor(half_excess)
                            off_bottom = ceil(half_excess)
                            # remove them
                            frame = frame[off_top:-off_bottom, :, :]
                    elif self.transform == "resize":
                        # resize change the aspect ratio
                        # this loses the aspect ratio
                        frame = cv2.resize(frame, (224, 224),
                                            interpolation=cv2.INTER_CUBIC)

                    # append the frame to the list
                    self.frames.append(frame)
                
                # read next frame
                ret, frame = vidcap.read()

            # check if the expected number of frames were read
            if self.numframe - (i_frame+1) <= 1:
                logging.debug("Video read properly")
                good_number_of_frames = True
            else:
                logging.debug("Video NOT read properly, adjusting fps and read again")
                self.fps_video = (i_frame+1) / self.time_second

        # convert frame from list to numpy array
        self.frames = np.array(self.frames)

    def __len__(self):
        """Return number of frames."""
        return len(self.frames)

    def __iter__(self, index):
        """Return frame at given index."""
        return self.frames[index]
    


class Frame:
    def __init__(self, video_path, FPS=2, transform=None, start=None, duration=None):

        self.FPS = FPS
        self.transform = transform

        # Knowing number of frames from FFMPEG metadata w/o without iterating over all frames
        videodata = skvideo.io.FFmpegReader(video_path)
        # numFrame x H x W x channels
        (numframe, _, _, _) = videodata.getShape()
        # if self.verbose:
            # print("shape video", videodata.getShape())
        self.time_second = _get_duration(video_path)
        # fps_video = numframe / time_second

        # time_second = getDuration(video_path)
        # if self.verbose:
        #     print("duration video", time_second)

        good_number_of_frames = False
        while not good_number_of_frames:
            fps_video = numframe / self.time_second
            # time_second = numframe / fps_video

            self.frames = []
            videodata = skvideo.io.vreader(video_path)
            # fps_desired = 2
            drop_extra_frames = fps_video/self.FPS

            # print(int(fps_video * start), int(fps_video * (start+45*60)))
            for i_frame, frame in tqdm(enumerate(videodata), total=numframe):
                # print(i_frame)

                for t in [0,5,10,15,20,25,30,35,40,45]:

                    if start is not None:
                        if i_frame == int(fps_video * (start + t*60)):
                        # print("saving image")
                            skvideo.io.vwrite(video_path.replace(".mkv", f"snap_{t}.png"), frame)
                            # os.path.join(os.path.dirname(video_path), f"snap_{t}.png"), frame)
                    # if i_frame == int(fps_video * (start+45*60)):
                    #     print("saving image")
                    #     skvideo.io.vwrite(os.path.join(os.path.dirname(video_path), "45.png"), frame)

                if start is not None:
                    if i_frame < fps_video * start:
                        continue

                if duration is not None:
                    if i_frame > fps_video * (start + duration):
                        # print("end of duration :)")
                        continue

                if (i_frame % drop_extra_frames < 1):

                    if self.transform == "resize256crop224":  # crop keep the central square of the frame
                        frame = imutils.resize(
                            frame, height=256)  # keep aspect ratio
                        # number of pixel to remove per side
                        off_side_h = int((frame.shape[0] - 224)/2)
                        off_side_w = int((frame.shape[1] - 224)/2)
                        frame = frame[off_side_h:-off_side_h,
                                        off_side_w:-off_side_w, :]  # remove them

                    elif self.transform == "crop":  # crop keep the central square of the frame
                        frame = imutils.resize(
                            frame, height=224)  # keep aspect ratio
                        # number of pixel to remove per side
                        off_side = int((frame.shape[1] - 224)/2)
                        frame = frame[:, off_side:-
                                        off_side, :]  # remove them

                    elif self.transform == "resize":  # resize change the aspect ratio
                        # lose aspect ratio
                        frame = cv2.resize(frame, (224, 224),
                                            interpolation=cv2.INTER_CUBIC)

                    # else:
                    #     raise NotImplmentedError()
                    # if self.array:
                    #     frame = img_to_array(frame)
                    self.frames.append(frame)

            print("expected number of frames", numframe,
                  "real number of available frames", i_frame+1)

            if numframe == i_frame+1:
                print("===>>> proper read! Proceeding! :)")
                good_number_of_frames = True
            else:
                print("===>>> not read properly... Read frames again! :(")
                numframe = i_frame+1
        
        self.frames = np.array(self.frames)

    def __len__(self):
        """Return number of frames."""
        return len(self.frames)

    def __iter__(self, index):
        """Return frame at given index."""
        return self.frames[index]


def _get_duration(video_path):
    """Get the duration (in seconds) for a video.

    Keyword arguments:
    video_path -- the path of the video
    """
    return moviepy.editor.VideoFileClip(video_path).duration
