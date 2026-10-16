// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::{Validate, ValidationError};

pub trait NvExtProvider {
    fn nvext(&self) -> Option<&NvExt>;
}

/// NVIDIA extensions to the OpenAI Videos API
#[derive(ToSchema, Serialize, Deserialize, Builder, Validate, Debug, Clone)]
#[validate(schema(function = "validate_nv_ext"))]
pub struct NvExt {
    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub annotations: Option<Vec<String>>,

    /// Frames per second (default: 24)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub fps: Option<i32>,

    /// Number of frames to generate (overrides fps * seconds if set)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub num_frames: Option<i32>,

    /// A text description of the undesired video content.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub negative_prompt: Option<String>,

    /// The number of denoising steps. More steps usually lead to higher quality at the expense of slower inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub num_inference_steps: Option<i32>,

    /// The CFG scale. Higher values usually lead to more coherent output.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub guidance_scale: Option<f32>,

    /// The seed for the random number generator.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub seed: Option<i64>,

    /// MoE expert switching boundary as a fraction of the denoising schedule (vLLM-Omni I2V).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub boundary_ratio: Option<f32>,

    /// CFG scale for the low-noise expert (vLLM-Omni I2V dual-guidance).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub guidance_scale_2: Option<f32>,

    /// Named output aspect ratio for video generation (e.g. "16:9", "1:1").
    /// Required by some models (e.g. MiniMax-H3 T2VA) when the pipeline cannot
    /// derive it from pixel dimensions alone; when provided it is forwarded to
    /// the diffusion sampling params as ``extra_args.aspect_ratio``.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub aspect_ratio: Option<String>,

    /// Per-keyframe semantic indices for FL2VA-style image-to-video generation.
    /// The list length must match the number of reference images and use
    /// model-native semantics (e.g. ``[0]`` for first frame, ``[-1]`` for last
    /// frame, ``[0, -1]`` for both). When ``input_references`` carries multiple
    /// images, the handler attaches them to ``multi_modal_data.image`` in order
    /// and writes this list to ``extra_args.frame_indices``.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub frame_indices: Option<Vec<i32>>,
}

impl Default for NvExt {
    fn default() -> Self {
        NvExt::builder().build().unwrap()
    }
}

impl NvExt {
    pub fn builder() -> NvExtBuilder {
        NvExtBuilder::default()
    }
}

fn validate_nv_ext(_nv_ext: &NvExt) -> Result<(), ValidationError> {
    Ok(())
}

impl NvExtBuilder {
    pub fn add_annotation(&mut self, annotation: impl Into<String>) -> &mut Self {
        self.annotations
            .get_or_insert_with(|| Some(vec![]))
            .as_mut()
            .expect("annotations should always be Some(Vec)")
            .push(annotation.into());
        self
    }
}
