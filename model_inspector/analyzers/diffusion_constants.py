# model_inspector/analyzers/diffusion_constants.py

from typing import Dict, Any, Tuple, List, Set
from ..models.confidence import ModelConfidence

# Shared constants for diffusion model analysis
DIFFUSION_MODEL_PATTERNS = {
    # Stable Diffusion families
    r'unet\.': ('UNet', ModelConfidence.HIGH),
    r'vae\.': ('VAE', ModelConfidence.HIGH),
    r'text_encoder\.': ('TextEncoder', ModelConfidence.HIGH),
    r'controlnet\.': ('ControlNet', ModelConfidence.HIGH),
    r'schedulers\.': ('Scheduler', ModelConfidence.HIGH),

    # Core SD patterns for main versions
    r'unet\.down_blocks\.0': ('StableDiffusion', ModelConfidence.MEDIUM),
    r'unet\.mid_block\.attentions': ('StableDiffusion', ModelConfidence.MEDIUM),

    # SD1.5 specific
    r'sd1\.5': ('StableDiffusion1_5', ModelConfidence.HIGH),
    r'sd-1\.5': ('StableDiffusion1_5', ModelConfidence.HIGH),
    r'sd-1-5': ('StableDiffusion1_5', ModelConfidence.HIGH),
    r'stable-diffusion-1-5': ('StableDiffusion1_5', ModelConfidence.HIGH),

    # SDXL specific
    r'text_encoder_2\.': ('SDXL', ModelConfidence.HIGH),
    r'transformer\.text_model\.encoder\.layers\.23\.': ('SDXL', ModelConfidence.HIGH),

    # SD2 specific
    r'text_model\.encoder\.layers\.11\.|text_model\.encoder\.layers\.12\.': ('SD2', ModelConfidence.MEDIUM),
    r'stable-diffusion-2': ('SD2', ModelConfidence.HIGH),
    r'sd-2': ('SD2', ModelConfidence.HIGH),
    r'sd2': ('SD2', ModelConfidence.HIGH),
    r'stable-diffusion-2-1': ('SD2_1', ModelConfidence.HIGH),
    r'sd-2-1': ('SD2_1', ModelConfidence.HIGH),
    r'sd2-1': ('SD2_1', ModelConfidence.HIGH),

    # SD3/SD3.5 patterns
    r'sd3\.': ('StableDiffusion3', ModelConfidence.HIGH),
    r'sd-3\.': ('StableDiffusion3', ModelConfidence.HIGH),
    r'stable-diffusion-3': ('StableDiffusion3', ModelConfidence.HIGH),
    r'transformer\.depth_layers\.23\.': ('StableDiffusion3', ModelConfidence.MEDIUM),
    r'transformer\.depth\.23\.': ('StableDiffusion3', ModelConfidence.MEDIUM),
    r'sd3-5\.': ('StableDiffusion3_5', ModelConfidence.HIGH),
    r'sd-3-5\.': ('StableDiffusion3_5', ModelConfidence.HIGH),
    r'sd-3\.5': ('StableDiffusion3_5', ModelConfidence.HIGH),
    r'stable-diffusion-3-5': ('StableDiffusion3_5', ModelConfidence.HIGH),
    r'stable-diffusion-3\.5': ('StableDiffusion3_5', ModelConfidence.HIGH),

    # Famous SD variants
    r'dreamshaper': ('Dreamshaper', ModelConfidence.HIGH),
    r'deliberate': ('Deliberate', ModelConfidence.HIGH),
    r'realvisxl': ('RealVisXL', ModelConfidence.HIGH),
    r'anything': ('AnythingV3', ModelConfidence.HIGH),
    r'anythingv3': ('AnythingV3', ModelConfidence.HIGH),
    r'anything-v3': ('AnythingV3', ModelConfidence.HIGH),
    r'anythingv4': ('AnythingV4', ModelConfidence.HIGH),
    r'anything-v4': ('AnythingV4', ModelConfidence.HIGH),
    r'anythingv5': ('AnythingV5', ModelConfidence.HIGH),
    r'anything-v5': ('AnythingV5', ModelConfidence.HIGH),
    r'openjourney': ('OpenJourney', ModelConfidence.HIGH),
    r'mdjrny': ('OpenJourney', ModelConfidence.HIGH),
    r'realisticvision': ('RealisticVision', ModelConfidence.HIGH),
    r'realistic_vision': ('RealisticVision', ModelConfidence.HIGH),
    r'juggernaut': ('Juggernaut', ModelConfidence.HIGH),
    r'epic_photogasm': ('EpicPhotogasm', ModelConfidence.HIGH),
    r'dreamlike': ('Dreamlike', ModelConfidence.HIGH),
    r'dreamlike-photoreal': ('DreamlikePhotoreal', ModelConfidence.HIGH),
    r'dreamlike-diffusion': ('DreamlikeDiffusion', ModelConfidence.HIGH),
    r'protogen': ('Protogen', ModelConfidence.HIGH),
    r'cyberrealistic': ('CyberRealistic', ModelConfidence.HIGH),
    r'cyber-realistic': ('CyberRealistic', ModelConfidence.HIGH),
    r'photoreal': ('PhotorealEngine', ModelConfidence.HIGH),
    r'photoreal-engine': ('PhotorealEngine', ModelConfidence.HIGH),
    r'majicmix': ('MajicMIX', ModelConfidence.HIGH),
    r'majic-mix': ('MajicMIX', ModelConfidence.HIGH),
    r'elldreth': ('Elldreth', ModelConfidence.HIGH),
    r'illuminati': ('Illuminati', ModelConfidence.HIGH),
    r'f222': ('F222', ModelConfidence.HIGH),
    r'timeless': ('Timeless', ModelConfidence.HIGH),
    r'neverending-dream': ('NeverendingDream', ModelConfidence.HIGH),
    r'ned': ('NeverendingDream', ModelConfidence.HIGH),
    r'analog': ('Analog', ModelConfidence.HIGH),
    r'lyriel': ('Lyriel', ModelConfidence.HIGH),
    r'samaritan': ('Samaritan', ModelConfidence.HIGH),
    r'edgeofreal': ('EdgeOfRealism', ModelConfidence.HIGH),
    r'edge-of-realism': ('EdgeOfRealism', ModelConfidence.HIGH),
    r'eor': ('EdgeOfRealism', ModelConfidence.MEDIUM),
    r'icbinp': ('ICBINP', ModelConfidence.HIGH),

    # Common anime-style models
    r'waifu-diffusion': ('WaifuDiffusion', ModelConfidence.HIGH),
    r'waifudiffusion': ('WaifuDiffusion', ModelConfidence.HIGH),
    r'nai': ('NovelAI', ModelConfidence.HIGH),
    r'novel.ai': ('NovelAI', ModelConfidence.HIGH),
    r'novelai': ('NovelAI', ModelConfidence.HIGH),
    r'abyss': ('AbyssOrangeMix', ModelConfidence.HIGH),
    r'aom': ('AbyssOrangeMix', ModelConfidence.HIGH),
    r'orangemix': ('AbyssOrangeMix', ModelConfidence.HIGH),
    r'hassanblend': ('HassanBlend', ModelConfidence.HIGH),
    r'chilloutmix': ('ChilloutMix', ModelConfidence.HIGH),
    r'counterfeit': ('Counterfeit', ModelConfidence.HIGH),
    r'meinaunreal': ('MeinaUnreal', ModelConfidence.HIGH),
    r'pony-diffusion': ('PonyDiffusion', ModelConfidence.HIGH),
    r'arcane-diffusion': ('ArcaneDiffusion', ModelConfidence.HIGH),

    # Newer variants
    r'redshift': ('Redshift', ModelConfidence.HIGH),
    r'colossus': ('Colossus', ModelConfidence.HIGH),
    r'animagine': ('Animagine', ModelConfidence.HIGH),
    r'reliberate': ('Reliberate', ModelConfidence.HIGH),
    r'rev': ('Rev', ModelConfidence.HIGH),
    r'ghostmix': ('GhostMix', ModelConfidence.HIGH),
    r'cuteyukimix': ('CuteYukiMix', ModelConfidence.HIGH),
    r'yuki': ('CuteYukiMix', ModelConfidence.HIGH),
    r'flat-2d': ('Flat2D', ModelConfidence.HIGH),
    r'flat2d': ('Flat2D', ModelConfidence.HIGH),
    r'meina': ('Meina', ModelConfidence.HIGH),
    r'pixelwave': ('PixelWave', ModelConfidence.HIGH),
    r'pastelboys': ('PastelBoys', ModelConfidence.HIGH),
    r'pastel': ('Pastel', ModelConfidence.HIGH),
    r'mo-di': ('MoDi', ModelConfidence.HIGH),
    r'distillery': ('Distillery', ModelConfidence.HIGH),
    r'nitro-diffusion': ('NitroDiffusion', ModelConfidence.HIGH),
    r'nitro': ('NitroDiffusion', ModelConfidence.MEDIUM),
    r'rinascita': ('Rinascita', ModelConfidence.HIGH),
    r'synthwave': ('Synthwave', ModelConfidence.HIGH),

    # LDM specific patterns
    r'ldm\.': ('LatentDiffusionModel', ModelConfidence.HIGH),

    # DiT patterns
    r'vision_encoder': ('DiT', ModelConfidence.MEDIUM),
    r'transformer\.blocks': ('DiT', ModelConfidence.MEDIUM),

    # Flux patterns
    r'flux_blocks': ('Flux', ModelConfidence.HIGH),
    r'model\.flux': ('Flux', ModelConfidence.HIGH),
    r'flux\.': ('Flux', ModelConfidence.HIGH),
    r'flux': ('Flux', ModelConfidence.HIGH),

    # Stable Cascade (Pony) patterns
    r'diffusion_model\.input_blocks\.11\.': ('StableCascade', ModelConfidence.HIGH),
    r'model\.pony': ('StableCascade_Pony', ModelConfidence.HIGH),
    r'stable_cascade_prior': ('StableCascade_Prior', ModelConfidence.HIGH),
    r'stable_cascade_decoder': ('StableCascade_Decoder', ModelConfidence.HIGH),
    r'pony': ('StableCascade_Pony', ModelConfidence.HIGH),
    r'ssd': ('StableCascade', ModelConfidence.HIGH),
    r'cascade': ('StableCascade', ModelConfidence.HIGH),
    r'illustrious': ('Illustrious', ModelConfidence.HIGH),

    # DeepFloyd IF patterns
    r'deepfloyd_if': ('DeepFloyd_IF', ModelConfidence.HIGH),
    r'if_unet': ('DeepFloyd_IF', ModelConfidence.HIGH),
    r'if_stage': ('DeepFloyd_IF', ModelConfidence.HIGH),

    # Kandinsky patterns
    r'kandinsky': ('Kandinsky', ModelConfidence.HIGH),
    r'movq\.': ('Kandinsky_MoVQ', ModelConfidence.HIGH),
    r'unet\.up\.3\.': ('Kandinsky', ModelConfidence.MEDIUM),

    # PixArt-α patterns
    r'pixart': ('PixArt', ModelConfidence.HIGH),
    r'pixart-sigma': ('PixArt_Sigma', ModelConfidence.HIGH),
    r'pixart-sigma-xl': ('PixArt_Sigma_XL', ModelConfidence.HIGH),
    r'pixart-α-xl': ('PixArt_Alpha_XL', ModelConfidence.HIGH),
    r'transformer\.blocks\.\d+\.': ('PixArt', ModelConfidence.MEDIUM),
    r'vision_hidden_size': ('PixArt', ModelConfidence.MEDIUM),

    # Playground models
    r'playground': ('Playground', ModelConfidence.HIGH),
    r'playground_v2': ('Playground_v2', ModelConfidence.HIGH),
    r'playground-v2.5': ('Playground_v2_5', ModelConfidence.HIGH),
    r'playground-v2-5': ('Playground_v2_5', ModelConfidence.HIGH),

    # LCM (Latent Consistency Models)
    r'lcm': ('LCM', ModelConfidence.HIGH),
    r'lcm_unet': ('LCM_UNet', ModelConfidence.HIGH),
    r'consistency_model': ('ConsistencyModel', ModelConfidence.HIGH),

    # DeciDiffusion
    r'deci': ('DeciDiffusion', ModelConfidence.HIGH),
    r'decidiffusion': ('DeciDiffusion', ModelConfidence.HIGH),

    # Muse/Midjourney-like models
    r'muse': ('Muse', ModelConfidence.HIGH),
    r'guided_diffusion': ('GuidedDiffusion', ModelConfidence.MEDIUM),
    r'midjourney': ('MidjourneyStyle', ModelConfidence.HIGH),
    r'dalle': ('DALLE', ModelConfidence.HIGH),
    r'dalle3': ('DALLE3', ModelConfidence.HIGH),
    r'unclip': ('UnCLIP', ModelConfidence.HIGH),

    # Wuerstchen patterns
    r'wuerstchen': ('Wuerstchen', ModelConfidence.HIGH),
    r'wurstchen': ('Wuerstchen', ModelConfidence.HIGH),  # Alternative spelling
    r'prior': ('DiffusionPrior', ModelConfidence.MEDIUM),
    r'decoder\.': ('DiffusionDecoder', ModelConfidence.MEDIUM),

    # Versatile Diffusion
    r'versatile_diffusion': ('VersatileDiffusion', ModelConfidence.HIGH),
    r'image_prefix_embedding': ('VersatileDiffusion', ModelConfidence.MEDIUM),

    # EMU (Facebook/Meta)
    r'emu': ('EMU', ModelConfidence.HIGH),
    r'emu_encoder': ('EMU_Encoder', ModelConfidence.HIGH),

    # SVD (Stable Video Diffusion)
    r'svd': ('StableVideoDiffusion', ModelConfidence.HIGH),
    r'temporal_transformer': ('StableVideoDiffusion', ModelConfidence.HIGH),
    r'motion_module': ('StableVideoDiffusion', ModelConfidence.HIGH),
    r'video': ('VideoModel', ModelConfidence.MEDIUM),
    r'videocrafter': ('VideoCrafter', ModelConfidence.HIGH),
    r'vdm': ('VideoDiffusionModel', ModelConfidence.HIGH),
    r'zeroscope': ('Zeroscope', ModelConfidence.HIGH),

    # Imagen (Google)
    r'imagen': ('Imagen', ModelConfidence.HIGH),
    r'imagen2': ('Imagen2', ModelConfidence.HIGH),
    r'imagen-2': ('Imagen2', ModelConfidence.HIGH),
    r'efficient_unet': ('Imagen', ModelConfidence.MEDIUM),

    # MagicVideo
    r'magicvideo': ('MagicVideo', ModelConfidence.HIGH),
    r'video_transformer': ('MagicVideo', ModelConfidence.MEDIUM),

    # SDXL Turbo
    r'sdxl_turbo': ('SDXL_Turbo', ModelConfidence.HIGH),
    r'sdxl-turbo': ('SDXL_Turbo', ModelConfidence.HIGH),
    r'sdxl_dpo': ('SDXL_DPO', ModelConfidence.HIGH),
    r'turbo': ('Turbo', ModelConfidence.MEDIUM),
    r'lightning': ('Lightning', ModelConfidence.HIGH),

    # Segmind models
    r'segmind': ('Segmind', ModelConfidence.HIGH),
    r'segmind_vega': ('Segmind_Vega', ModelConfidence.HIGH),
    r'ssd_1b': ('Segmind_SSD', ModelConfidence.HIGH),
    r'vega': ('Segmind_Vega', ModelConfidence.HIGH),
    r'damo': ('DAMO', ModelConfidence.HIGH),

    # AnimateDiff
    r'animatediff': ('AnimateDiff', ModelConfidence.HIGH),
    r'motion_module': ('AnimateDiff', ModelConfidence.HIGH),

    # Diffusers suite
    r'diffusers\.': ('DiffusersModel', ModelConfidence.HIGH),

    # Adapters
    r'adapter\.': ('T2IAdapter', ModelConfidence.HIGH),
    r'ip_adapter\.': ('IPAdapter', ModelConfidence.HIGH),
    r'ip_adapter_image_proj': ('IPAdapter', ModelConfidence.HIGH),
    r'lora_unet': ('LoRA', ModelConfidence.HIGH),
    r'lora_te': ('LoRA_TextEncoder', ModelConfidence.HIGH),

    # Control models
    r'controlnet_': ('ControlNet', ModelConfidence.HIGH),
    r'controlnet_canny': ('ControlNet_Canny', ModelConfidence.HIGH),
    r'controlnet_depth': ('ControlNet_Depth', ModelConfidence.HIGH),
    r'controlnet_pose': ('ControlNet_Pose', ModelConfidence.HIGH),
    r'controlnet_seg': ('ControlNet_Segmentation', ModelConfidence.HIGH),
    r'controlnet_tile': ('ControlNet_Tile', ModelConfidence.HIGH),

    # ReVision models
    r'revision': ('ReVision', ModelConfidence.HIGH),
    r'image_conditioner': ('ReVision', ModelConfidence.MEDIUM),

    # Specialized variants
    r'inpainting': ('Inpainting', ModelConfidence.HIGH),
    r'upscaler': ('Upscaler', ModelConfidence.HIGH),
    r'pix2pix': ('Pix2Pix', ModelConfidence.HIGH),
    r'dreambooth': ('DreamBooth', ModelConfidence.MEDIUM),
    r'subject_driven': ('SubjectDriven', ModelConfidence.MEDIUM),

    # Textual embedding patterns
    r'<.*>': ('TextualEmbedding', ModelConfidence.HIGH),
    r'emb_params': ('TextualEmbedding', ModelConfidence.HIGH),
    r'string_to_param': ('TextualEmbedding', ModelConfidence.HIGH),
    r'name_to_idx': ('TextualEmbedding', ModelConfidence.HIGH),
    r'learned_embeds': ('TextualEmbedding', ModelConfidence.HIGH),
    r'clip_l': ('TextualEmbedding', ModelConfidence.HIGH),
}

# Model configuration hints
DIFFUSION_CONFIG_HINTS = {
    # Diffusion model types
    'model_type': {
        'unet': ('UNet', ModelConfidence.HIGH),
        'vae': ('VAE', ModelConfidence.HIGH),
        'clip': ('CLIPTextEncoder', ModelConfidence.HIGH),
        'controlnet': ('ControlNet', ModelConfidence.HIGH),
        'adapter': ('T2IAdapter', ModelConfidence.HIGH),
        'ip_adapter': ('IPAdapter', ModelConfidence.HIGH),
        'prior': ('DiffusionPrior', ModelConfidence.HIGH),
        'decoder': ('DiffusionDecoder', ModelConfidence.HIGH),
    },
    # Architecture types
    'architectures': {
        'UNet2DConditionModel': ('StableDiffusion_UNet', ModelConfidence.HIGH),
        'AutoencoderKL': ('StableDiffusion_VAE', ModelConfidence.HIGH),
        'CLIPTextModel': ('StableDiffusion_TextEncoder', ModelConfidence.HIGH),
        'ControlNetModel': ('ControlNet', ModelConfidence.HIGH),
        'StableCascadeUNet': ('StableCascade_UNet', ModelConfidence.HIGH),
        'StableCascadePriorPipeline': ('StableCascade_Prior', ModelConfidence.HIGH),
        'StableCascadeDecoderPipeline': ('StableCascade_Decoder', ModelConfidence.HIGH),
        'StableCascadeDecoderModel': ('StableCascade_Decoder', ModelConfidence.HIGH),
        'PriorTransformer': ('DiffusionPrior', ModelConfidence.HIGH),
        'VQModel': ('VQGANModel', ModelConfidence.HIGH),
        'T2IAdapter': ('T2IAdapter', ModelConfidence.HIGH),
        'IPAdapterModel': ('IPAdapter', ModelConfidence.HIGH),
        'FluxModel': ('Flux', ModelConfidence.HIGH),
        'DeepFloydIFPipeline': ('DeepFloyd_IF', ModelConfidence.HIGH),
        'KandinskyCombinedPipeline': ('Kandinsky', ModelConfidence.HIGH),
        'PixArtAlphaPipeline': ('PixArt', ModelConfidence.HIGH),
        'TransformerTemporalModel': ('StableVideoDiffusion', ModelConfidence.HIGH),
        'ConsistencyModelPipeline': ('ConsistencyModel', ModelConfidence.HIGH),
        'LatentConsistencyModelPipeline': ('LCM', ModelConfidence.HIGH),
        'AnimateDiffPipeline': ('AnimateDiff', ModelConfidence.HIGH),
        'MotionAdapter': ('AnimateDiff_Adapter', ModelConfidence.HIGH),
        'EmuPipeline': ('EMU', ModelConfidence.HIGH),
        'ImageInpaintingPipeline': ('Inpainting', ModelConfidence.HIGH),
        'Pix2PixModel': ('Pix2Pix', ModelConfidence.HIGH),
        'WuerstchenPrior': ('Wuerstchen_Prior', ModelConfidence.HIGH),
        'WuerstchenDecoder': ('Wuerstchen_Decoder', ModelConfidence.HIGH),
        'IllustriousPriorPipeline': ('Illustrious_Prior', ModelConfidence.HIGH),
        'IllustriousDecoderPipeline': ('Illustrious_Decoder', ModelConfidence.HIGH),
    },
    # SD versions based on cross attention dimensions
    'cross_attention_dim': {
        768: ('SD1.x_Component', ModelConfidence.MEDIUM),
        1024: ('SD2.x_Component', ModelConfidence.MEDIUM),
        1280: ('SDXL_Component', ModelConfidence.MEDIUM),
        2048: ('SDXL_Component', ModelConfidence.MEDIUM),
    },
    # Specific model identification via prediction_type
    'prediction_type': {
        'epsilon': ('EpsilonModel', ModelConfidence.LOW),
        'v_prediction': ('V-PredictionModel', ModelConfidence.LOW),
        'sample': ('SamplePredictionModel', ModelConfidence.LOW),
        'velocity': ('VelocityModel', ModelConfidence.LOW),
    },
    # Identify by pipeline name
    'pipeline_class_name': {
        'StableDiffusionPipeline': ('StableDiffusion', ModelConfidence.HIGH),
        'StableDiffusionInpaintPipeline': ('StableDiffusion_Inpainting', ModelConfidence.HIGH),
        'StableDiffusionXLPipeline': ('SDXL', ModelConfidence.HIGH),
        'StableDiffusionXLInpaintPipeline': ('SDXL_Inpainting', ModelConfidence.HIGH),
        'StableDiffusion3Pipeline': ('StableDiffusion3', ModelConfidence.HIGH),
        'StableDiffusion3ImgToImgPipeline': ('StableDiffusion3', ModelConfidence.HIGH),
        'StableDiffusion3InpaintPipeline': ('StableDiffusion3_Inpainting', ModelConfidence.HIGH),
        'StableDiffusion35Pipeline': ('StableDiffusion3_5', ModelConfidence.HIGH),
        'StableDiffusion35ImgToImgPipeline': ('StableDiffusion3_5', ModelConfidence.HIGH),
        'StableDiffusion35InpaintPipeline': ('StableDiffusion3_5_Inpainting', ModelConfidence.HIGH),
        'FluxPipeline': ('Flux', ModelConfidence.HIGH),
        'StableCascadePipeline': ('StableCascade', ModelConfidence.HIGH),
        'StableCascadeCombinedPipeline': ('StableCascade', ModelConfidence.HIGH),
        'IllustriousPipeline': ('Illustrious', ModelConfidence.HIGH),
        'DeepFloydIFPipeline': ('DeepFloyd_IF', ModelConfidence.HIGH),
        'KandinskyPipeline': ('Kandinsky', ModelConfidence.HIGH),
        'Kandinsky2Pipeline': ('Kandinsky2', ModelConfidence.HIGH),
        'PixArtAlphaPipeline': ('PixArt', ModelConfidence.HIGH),
        'LatentConsistencyModelPipeline': ('LCM', ModelConfidence.HIGH),
        'StableVideoDiffusionPipeline': ('StableVideoDiffusion', ModelConfidence.HIGH),
        'AnimateDiffPipeline': ('AnimateDiff', ModelConfidence.HIGH),
        'WuerstchenPipeline': ('Wuerstchen', ModelConfidence.HIGH),
        'VersatileDiffusionPipeline': ('VersatileDiffusion', ModelConfidence.HIGH),
    },
}

# Special pattern to identify LoRA adapters
LORA_PATTERNS = [
    r'lora_unet_',
    r'lora_te_',
    r'lora_down_',
    r'lora_up_',
    r'lora_mid_',
    r'\.alpha',
    r'\.weight',
    r'\.bias',
    r'\.scale',
    r'\.diff',
    r'^up\.',
    r'^down\.',
    r'^alphas'
]

# Patterns for identifying trigger words in textual inversions/embedding models
EMBEDDING_TRIGGER_PATTERNS = [
    r'<(?P<token>[^>]+)>',
    r'token_(?P<token>\w+)',
    r'embedding_(?P<token>\w+)',
    r'name_to_token',
    r'string_to_token',
    r'tokens_list',
    r'ss_token_string',
    r'learned_embed_'
]

# Patterns for identifying trigger words in LoRA models
LORA_TRIGGER_PATTERNS = [
    r'ss_activation_message',
    r'ss_activation_text',
    r'ss_training_prompt',
    r'ss_trigger_string',
    r'activation_text',
    r'ss_trigger_keyword',
    r'activation_word',
    r'trigger'
]

# Common configuration field names in diffusers models
CONFIG_FIELDS = [
    '_class_name',
    '_diffusers_version',
    'architectures',
    'model_type',
    'cross_attention_dim',
    'sample_size',
    'in_channels',
    'out_channels',
    'down_block_types',
    'up_block_types',
    'projection_class_embeddings_input_dim',
    'prediction_type',
    'time_embedding_type',
    'timestep_spacing',
    'transformer_layers_per_block',
    'norm_num_groups',
    'conditioning_embedding_out_channels',
    'controlnet_conditioning_channel_order',
    'pipeline_class_name',
    'vae_scaling_factor',
    'image_processor',
    'text_encoder',
    'unet',
    'scheduler',
    'hidden_size',
    'pooling_mode',
    'max_position_embeddings',
    'vocab_size',
    'classifier_dropout',
    'type_vocab_size',
    'embedding_size',
    'similarity_metric',
    'normalize_embeddings',
]

# Map of metadata fields that may contain trigger words
TRIGGER_WORD_FIELDS = [
    'ss_trigger_word',
    'ss_activation_text',
    'ss_training_prompt',
    'activation_text',
    'trigger_keyword',
    'trigger',
    'embedding_name',
    'token_string',
    'training_prompt',
    'train_text_encoder_words',
    'train_text_encoder_params',
    'ss_tag_frequency',
    'ss_concepts_list'
]
