#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import librosa


def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

def print_metrics(perf_metrics):
    print(f"\n\nLoad time: {perf_metrics.get_load_time():.2f} ms")
    print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("wav_file_path")
    args = parser.parse_args()

    device = "CPU"  # GPU, NPU can be used as well
    pipe = openvino_genai.WhisperPipeline(args.model_dir, device)

    config = pipe.get_generation_config()
    config.max_new_tokens = 100  # increase this based on your speech length
    # 'task' and 'language' parameters are supported for multilingual models only
    config.language = "<|en|>"  # can switch to <|zh|> for Chinese language
    config.task = "transcribe"
    config.return_timestamps = True

    # Pipeline expects normalized audio with Sample Rate of 16kHz
    raw_speech = read_wav(args.wav_file_path)
    result = pipe.generate(raw_speech, config)

    print(result)

    if result.chunks:
        for chunk in result.chunks:
            print(f"timestamps: [{chunk.start_ts:.2f}, {chunk.end_ts:.2f}] text: {chunk.text}")

    print_metrics(result.perf_metrics)


if "__main__" == __name__:
    main()
