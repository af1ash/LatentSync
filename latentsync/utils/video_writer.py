#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 2026/02/06
Author: Lixiaobing
Desc: 视频编码及推流
"""

import av

from fractions import Fraction


class VideoWriter:
    """
    codec:
        prores_ks: yuv422p10le yuv444p10le yuva444p10le
    """
    vformat2codec = {
        # pix_fmt, codec, audio_codec, options
        ".mov": [
            "yuv422p10le",
            "prores_ks",
            "aac",
            {
                "tune": "zerolatency",
                'profile': '3',  # ProRes 422 HQ
                'vendor': 'apl0',
                'qscale': '10',   # 质量参数（可选）
            },
        ],
        # ".mov": [
        #     "yuv444p10le",
        #     "prores_ks",
        #     "aac",
        #     {
        #         "tune": "zerolatency",
        #     },
        # ],
        ".webm": [
            "yuva420p",
            "libvpx-vp9",
            "libvorbis",
            {
                "deadline": "realtime",
            },
        ],
        ".mkv": [
            "yuva420p",
            "ffv1",
            "aac",
            {
                "tune": "zerolatency",
            },
        ],
        ".mp4": [
            "yuv420p",
            "libx264",
            "aac",
            {
                "tune": "zerolatency",
            },
        ],
        ".flv": [
            "yuv420p",
            "h264",
            "aac",
            {
                "tune": "zerolatency",
            },
        ],
    }

    def __init__(
        self,
        source_url,
        outformat,
    ):
        self.source_url = source_url

        self.container = av.open(self.source_url, mode="w")
        self.pix_fmt, self.codec, self.audio_codec, self.default_options = (
            self.vformat2codec[outformat]
        )
        self.audio_stream = None

    def init_stream(self, frame_data, audio_data, fps, sample_rate):
        # 视频流配置
        height, width, vchannels = frame_data.shape
        self.video_stream = self.container.add_stream(self.codec, rate=fps)
        self.video_stream.width = width
        self.video_stream.height = height
        self.video_stream.pix_fmt = self.pix_fmt  # 常用像素格式
        self.video_stream.time_base = Fraction(1, fps)
        self.video_stream.options = {
            "preset": "ultrafast",
            "crf": "23",
            "g": str(fps),  # GOP = 1秒
            "keyint_min": str(fps),
            "x264opts": "repeat-headers=1",  # 👈 关键：重复 SPS/PPS
        }
        self.video_stream.options.update(self.default_options)

        if vchannels == 4:
            self.vformat = "rgba"
        else:
            self.vformat = "rgb24"

        # 音频流配置
        if audio_data is not None:
            achannels, audio_chunk_size = audio_data.shape
            self.audio_chunk_size = audio_chunk_size
            if achannels == 2:
                self.audio_layout = "stereo"
            else:
                self.audio_layout = "mono"
            self.audio_stream = self.container.add_stream(
                self.audio_codec,
                rate=sample_rate,
                layout=self.audio_layout,
            )
            self.audio_format = "fltp"
            self.sample_rate = sample_rate
            self.audio_stream.time_base = Fraction(1, sample_rate)
            # self.audio_stream.bit_rate = sample_rate
            self.audio_stream.codec_context.options = {
                "aac_adtstoasc": "1"  # 这个选项在 muxing 时会生成正确的 ASC
            }

    def encode_frame(self, frame_data, audio_data, video_pts):
        # print(f"{frame_data.shape=},{audio_data.shape=}")
        # 创建 AVFrame
        print(f"{self.vformat=}")
        av_frame = av.VideoFrame.from_ndarray(frame_data, format=self.vformat)
        # 设置时间戳
        av_frame.pts = video_pts
        # 编码帧
        for packet in self.video_stream.encode(av_frame):
            self.container.mux(packet)

        if audio_data is not None:
            audio_avframe = av.AudioFrame.from_ndarray(
                audio_data,
                format=self.audio_format,
                layout=self.audio_layout,
            )
            audio_avframe.sample_rate = self.sample_rate
            audio_avframe.pts = video_pts * self.audio_chunk_size
            # 编码并写入音频包
            for packet in self.audio_stream.encode(audio_avframe):
                self.container.mux(packet)

    def close(self):
        # 冲洗编码器（flush）
        for packet in self.video_stream.encode():
            self.container.mux(packet)
        # Flush audio stream
        if self.audio_stream:
            for packet in self.audio_stream.encode():
                self.container.mux(packet)
        # 关闭容器
        return self.container.close()

    def write(
        self,
        static_generator,
        fps=25,
        sample_rate=16000,
    ):
        for video_pts, frame in enumerate(static_generator):
            frame_data = frame["video"]
            audio_data = frame.get("audio")

            if video_pts == 0:
                # 视频流配置
                height, width, vchannels = frame_data.shape
                self.video_stream = self.container.add_stream(
                    self.codec, rate=fps
                )
                self.video_stream.width = width
                self.video_stream.height = height
                self.video_stream.pix_fmt = self.pix_fmt  # 常用像素格式
                self.video_stream.time_base = Fraction(1, fps)
                self.video_stream.options = {
                    "preset": "ultrafast",
                    "tune": "zerolatency",
                    "crf": "23",
                    "g": str(fps),  # GOP = 1秒
                    "keyint_min": str(fps),
                    "x264opts": "repeat-headers=1",  # 👈 关键：重复 SPS/PPS
                }

                if vchannels == 4:
                    self.vformat = "rgba"
                else:
                    self.vformat = "rgb24"

                # 音频流配置
                if audio_data is not None:
                    achannels, audio_chunk_size = audio_data.shape
                    self.audio_chunk_size = audio_chunk_size
                    if achannels == 2:
                        self.audio_layout = "stereo"
                    else:
                        self.audio_layout = "mono"
                    self.audio_stream = self.container.add_stream(
                        self.audio_codec,
                        rate=sample_rate,
                        layout=self.audio_layout,
                    )
                    self.audio_format = "fltp"
                    self.audio_stream.time_base = Fraction(1, sample_rate)
                    self.audio_stream.codec_context.options = {
                        "aac_adtstoasc": "1"  # 这个选项在 muxing 时会生成正确的 ASC
                    }

            # 创建 AVFrame
            av_frame = av.VideoFrame.from_ndarray(
                frame_data, format=self.vformat
            )
            # 设置时间戳
            av_frame.pts = video_pts
            # 编码帧
            for packet in self.video_stream.encode(av_frame):
                self.container.mux(packet)

            if audio_data:
                audio_avframe = av.AudioFrame.from_ndarray(
                    audio_data,
                    format=self.audio_format,
                    layout=self.audio_layout,
                )
                audio_avframe.sample_rate = sample_rate
                audio_avframe.pts = video_pts * self.audio_chunk_size
                # 编码并写入音频包
                for packet in self.audio_stream.encode(audio_avframe):
                    self.container.mux(packet)

        # 冲洗编码器（flush）
        for packet in self.video_stream.encode():
            self.container.mux(packet)
        # Flush audio stream
        if audio_data is None:
            for packet in self.audio_stream.encode():
                self.container.mux(packet)
        # 关闭容器
        return self.container.close()


class VideoReader:
    def __init__(self, input_videoname):
        self.input_videoname = input_videoname

    @property
    def average_rate(self):
        with av.open(self.input_videoname) as container:
            video_stream = container.streams.video[0]
            return video_stream.average_rate

    @property
    def frames(self):
        with av.open(self.input_videoname) as container:
            video_stream = container.streams.video[0]
            return video_stream.frames

    def read(self, format="bgra"):
        frame_list = []

        with av.open(self.input_videoname) as container:
            for _, frame in enumerate(container.decode(video=0)):
                # 转换为 RGBA（保留 Alpha）
                img = frame.to_ndarray(format=format)
                frame_list.append(img)
        return frame_list

    def read_iter(self, format="bgra"):
        with av.open(self.input_videoname) as container:
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format=format)
                yield img
