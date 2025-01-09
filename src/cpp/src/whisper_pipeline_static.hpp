// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "whisper/whisper_models.hpp"
#include "whisper_pipeline_base.hpp"

namespace ov {
namespace genai {

class DecoderCache {
public:
    DecoderCache() = default;
    DecoderCache(std::shared_ptr<ov::Model> model, ov::PartialShape shape)
     : m_decoder_model(model)
     , m_lhs_shape(shape) {}

    ov::CompiledModel get_model(uint8_t input_ids_size);
private:
    std::unordered_map<uint8_t, ov::CompiledModel> m_cache;
    std::shared_ptr<ov::Model> m_decoder_model;
    ov::PartialShape m_lhs_shape;
};

class WhisperPipeline::StaticWhisperPipeline : public WhisperPipeline::WhisperPipelineImplBase {
public:
    StaticWhisperPipeline(const std::filesystem::path& model_path, const ov::AnyMap& properties);

    WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input,
                                   OptionalWhisperGenerationConfig generation_config,
                                   ChunkStreamerVariant streamer) override;

private:
    WhisperInitializedModels m_models;
    DecoderCache m_decoder_cache;
};

}  // namespace genai
}  // namespace ov
