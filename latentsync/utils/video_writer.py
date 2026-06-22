#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 2026/02/06
Author: Lixiaobing
Desc: 视频编码及推流
"""

import av
import numpy as np
from pathlib import Path

from fractions import Fraction


class VideoWriter:
    """
    codec:
        prores_ks: yuv422p10le yuv444p10le yuva444p10le
    """
    vformat2codec = {
        # codec:
        #     prores_ks: yuv422p10le yuv444p10le yuva444p10le
        # container, pix_fmt, codec, audio_codec, options
        "mov": [
            "mov",
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
        "mov_alpha": [
            "mov",
            "yuva444p10le",
            "prores_ks",
            "aac",
            {
                "tune": "zerolatency",
            },
        ],
        "webm": [
            "webm",
            "yuva420p",
            "libvpx-vp9",
            "libvorbis",
            {
                "deadline": "realtime",
            },
        ],
        "mkv": [
            "mkv",
            "yuva420p",
            "ffv1",
            "aac",
            {
                "tune": "zerolatency",
            },
        ],
        "mp4": [
            "mp4"
            "yuv420p",
            "libx264",
            "aac",
            {
                "tune": "zerolatency",
            },
        ],
        "flv": [
            "mp4"
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
        codec_info,
    ):
        self.source_url = source_url

        self.container = av.open(self.source_url, mode="w")
        # self.pix_fmt, self.codec, self.audio_codec, self.default_options = (
        #     self.vformat2codec[outformat]
        # )
        (
            self.container_fmt,
            self.pix_fmt,
            self.codec,
            self.audio_codec,
            self.default_options,
        ) = self.vformat2codec[codec_info]

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
        # print(f"{self.vformat=}")

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
        # print(f"{self.vformat=}")
        av_frame = av.VideoFrame.from_ndarray(frame_data, format=self.vformat)
        # 设置时间戳
        av_frame.pts = video_pts
        # 编码帧
        for packet in self.video_stream.encode(av_frame):
            self.container.mux(packet)

        if audio_data is not None and self.audio_stream:
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
        self.container = av.open(self.input_videoname)
        # suffix = Path(self.input_videoname).suffix
        # if suffix in ['.mov', '.mkv']:
        #     self.vformat = 'rgba'
        # else:
        #     self.vformat = 'rgb24'
        self.audio_stream = None
        self.video_stream = None
        for stream in self.container.streams:
            if stream.type == "audio":
                self.audio_stream = stream
            elif stream.type == "video":
                self.video_stream = stream
                if "a" in self.video_stream.pix_fmt:
                    self.vformat = "rgba"
                else:
                    self.vformat = "rgb24"

    @property
    def fps(self):
        return int(self.video_stream.average_rate)

    @property
    def frames(self):
        if self.video_stream.frames > 0:
            return self.video_stream.frames
        elif self.container.duration > 0:
            return int(
                self.container.duration
                / 1000
                / 1000
                * int(self.video_stream.average_rate)
            )
        else:
            return self.video_stream.frames

    @property
    def sample_rate(self):
        if self.audio_stream:
            return self.audio_stream.rate
        return None

    @property
    def channels(self):
        if self.audio_stream:
            return self.audio_stream.layout.channels
        return None

    @property
    def layout(self):
        if self.audio_stream:
            return self.audio_stream.layout.name
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.container.close()

    def read(self, format="bgra", type_="video"):
        frame_list = []
        for frame in self.read_iter(format, type_):
            frame_list.append(frame)
        if type_ == "audio":
            return np.concatenate(frame_list, axis=1)
        return np.array(frame_list)

    def read_iter(self, format="rgba", type_="video"):
        if type_ == "video":
            read_stream = self.video_stream
            frame_kwargs = {"format": self.vformat}
        elif type_ == "audio":
            read_stream = self.audio_stream
            frame_kwargs = {}
        else:
            raise ValueError(f"unknown {type_=}")
        self.container.seek(0)
        for frame in self.container.decode(read_stream):
            data = frame.to_ndarray(**frame_kwargs)
            # data 的维度取决于格式：
            # 如果是平面格式 (如 fltp, s16p): shape 为 (通道数, 采样点数)
            # 如果是交错格式 (如 flt, s16): shape 为 (1, 采样点数 * 通道数)
            if type_ == "audio":
                if self.audio_stream.channels == 2:
                    if "p" in self.audio_stream.format.name:
                        # 平面格式：直接获取左右声道
                        left_channel = data[0]
                        right_channel = data[1]
                    else:
                        # 交错格式：需要切片分离 [L, R, L, R, ...]
                        left_channel = data[0, ::2]
                        right_channel = data[0, 1::2]
                    data = np.array([left_channel, right_channel])
            # print(f"{data.shape=}")
            yield data

    def close(self):
        return self.container.close()
