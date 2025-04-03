"""
    Analyzer for safetensors format files.

    Safetensors is a binary format for storing tensors and associated metadata in
    a safe and efficient manner. This analyzer extracts metadata from safetensors
    files to identify the model type, architecture, and other relevant parameters.

    This module supports detection of various model architectures including language
    models, vision models, audio models, embeddings, and adapter formats like LoRA.
    It deliberately avoids diffusion model detection.
"""
from typing import Dict, Any, Tuple, Optional, List, Set
import json
import re
import logging
from pathlib import Path
import struct
from collections import Counter

from ..models.confidence import ModelConfidence
from .base import BaseAnalyzer


class SafetensorsAnalyzer(BaseAnalyzer):
    """
    # SafetensorsAnalyzer: Limitations and Improvement Opportunities

    ## Current Limitations

    ### 1. Pattern-Based Model Detection

    - **Brittleness to Naming Variations**: The pattern matching approach is sensitive to tensor naming conventions, which can vary across frameworks, model versions, and custom implementations. Small changes in naming patterns can cause models to be misclassified or remain unidentified.

    - **Limited Contextual Understanding**: Pattern matching examines each tensor name in isolation without considering relationships between tensors, resulting in potential misclassifications when similar patterns appear across different architectures.

    - **Manual Pattern Maintenance**: The `MODEL_PATTERNS` dictionary requires manual curation and updates as new model architectures emerge, creating maintenance overhead and inevitable gaps in coverage.

    ### 2. Metadata Parsing

    - **Inconsistent Metadata Standards**: The safetensors format doesn't enforce a standard schema for metadata, leading to inconsistencies in how model information is stored across different sources.

    - **Limited JSON Field Handling**: Only predefined JSON fields are parsed (`JSON_FIELDS` set), potentially missing important structured data stored in other fields.

    - **Metadata Type Ambiguity**: The analyzer doesn't handle type validation or normalization of metadata fields, potentially leading to inconsistent representations of similar information.

    ### 3. Model Dimension Inference

    - **Simplistic Dimension Heuristics**: The current approach uses individual tensor dimensions without considering architectural context, potentially leading to false associations with specific model types.

    - **Incomplete Dimension Coverage**: The `DIMENSION_HINTS` dictionary only covers a limited set of common dimensions, missing many valid model configurations.

    - **Limited Layer Pattern Recognition**: The layer counting logic only works with a predefined set of patterns and can fail with custom or unusual naming schemes.

    ### 4. Architectural Analysis

    - **Surface-Level Feature Detection**: The feature detection looks for keywords rather than analyzing the actual structure of the model, leading to potential false positives.

    - **Limited Component Inference**: The component analysis doesn't determine the relationships between components or their hierarchical organization in the model architecture.

    - **Minimal Quantitative Analysis**: Beyond parameter counting, there's limited analysis of computational complexity, memory footprint, or runtime performance characteristics.

    ### 5. Manual Parsing Fallback

    - **Simplified Header Parsing**: The manual parsing fallback provides basic header extraction but doesn't fully replicate the functionality of the safetensors library.

    - **No Tensor Data Validation**: The fallback approach doesn't validate tensor metadata against actual data sizes or offsets, potentially missing corrupted files.

    ### 6. Model Type Classification

    - **Coarse Classification Granularity**: Many models are classified into broad categories (e.g., "LanguageModel") rather than specific architectures or variants.

    - **Limited Confidence Calibration**: The confidence scores are assigned heuristically and may not accurately reflect the true certainty of model type detection.

    - **Insufficient Handling of Hybrid Models**: The analyzer struggles with multi-task or hybrid architectures that combine elements from different model families.

    ## Improvement Opportunities

    ### 1. Enhanced Model Detection Architecture

    - **Hierarchical Taxonomy System**: Replace the flat pattern matching with a hierarchical taxonomy of model architectures, allowing for more granular classification (e.g., Transformer → Language Model → Decoder-only → GPT-family → GPT-2).

    - **Graph-based Structure Analysis**: Implement analysis of the tensor graph structure to identify common architectural patterns regardless of naming conventions.

    - **Statistical Signature Matching**: Create statistical "signatures" of known model architectures based on tensor shapes, counts, and relationships that can identify models even with renamed tensors.

    ### 2. Improved Metadata Processing

    - **Schema Inference and Validation**: Implement dynamic schema inference to handle various metadata formats and validate against known schemas where possible.

    - **Metadata Normalization**: Create a standardized internal representation of metadata to ensure consistent outputs regardless of input format variations.

    - **Intelligent Field Type Conversion**: Implement more sophisticated type detection and conversion for metadata fields (e.g., recognizing date formats, version strings, or numeric ranges).

    ### 3. Advanced Dimension Analysis

    - **Dimensional Relationship Analysis**: Analyze relationships between tensor dimensions (such as hidden size to attention head ratios) to better identify model architectures.

    - **Layer Grouping and Analysis**: Group tensors by layers and analyze layer-wise patterns to understand the model's depth, width, and architectural features.

    - **Dynamic Dimension Mapping**: Implement a learning system that can adapt to new dimension patterns based on verified model identifications.

    ### 4. Deeper Architectural Insights

    - **Computational Graph Reconstruction**: Attempt to reconstruct the computational graph from tensor names and shapes to provide more accurate architecture visualization.

    - **Performance Characteristic Estimation**: Estimate computational requirements, memory usage, and potential bottlenecks based on tensor analysis.

    - **Architectural Comparison**: Provide comparison metrics to similar known architectures to help users understand model relationships.

    ### 5. Technical Enhancements

    - **Improved Manual Parsing**: Enhance the fallback parsing method to handle more file format variations and validate tensor metadata more thoroughly.

    - **Streaming Processing**: Implement streaming support for large files to avoid loading entire headers into memory.

    - **Parallel Processing**: Implement parallel processing for analyzing multiple tensors simultaneously in very large models.

    ### 6. Integration and Extension

    - **Framework-Specific Optimizations**: Add specialized detection logic for common frameworks (PyTorch, TensorFlow, JAX) to leverage framework-specific patterns.

    - **Versioning Analysis**: Add capabilities to detect version differences between similar model architectures.

    - **Cross-Format Compatibility**: Extend the analyzer to understand relationships between safetensors and other model formats (GGUF, ONNX, etc.) for the same models.

    ### 7. Machine Learning-Based Improvements

    - **Model Fingerprinting**: Develop a machine learning approach to "fingerprint" known model architectures based on their tensor distribution and relationship patterns.

    - **Automated Pattern Discovery**: Use clustering and pattern recognition to automatically discover new model architectures from a corpus of safetensors files.

    - **Confidence Calibration**: Train a model to provide well-calibrated confidence scores for model type predictions based on multiple features.

    ### 8. User Experience Enhancements

    - **Rich Report Generation**: Implement detailed report generation with visualizations of model architecture and statistics.

    - **Interactive Exploration**: Provide an interactive mode for exploring model details, allowing users to drill down into specific components.

    - **Compatibility Information**: Include information about framework compatibility and conversion requirements based on model architecture.

    ## Implementation Priority Recommendations

    1. **Develop a comprehensive taxonomy** of model architectures to replace the simple pattern matching approach
    2. **Enhance dimension analysis** to consider relationships between tensors
    3. **Implement better metadata normalization** for consistent outputs
    4. **Add layer grouping and analysis** for better architectural understanding
    5. **Improve confidence scoring** with multiple heuristics combined

    These improvements would significantly enhance the analyzer's accuracy, robustness, and utility for a wide range of model types while maintaining its focused purpose of non-diffusion model analysis.
    """

    # JSON fields that might contain stringified JSON
    JSON_FIELDS = {'ss_metadata', 'train_params', 'network_args', 'modelspec.config'}

    # Model type patterns in tensor names with associated confidence
    MODEL_PATTERNS = {
        # Language model patterns
        r'model\.layers\.\d+\.self_attn': ('LanguageModel', ModelConfidence.MEDIUM),
        r'llama\.layers': ('LLaMA', ModelConfidence.HIGH),
        r'mistral\.layers': ('Mistral', ModelConfidence.HIGH),
        r'attention\.wq\.weight': ('LanguageModel', ModelConfidence.MEDIUM),
        r'model\.embed_tokens\.weight': ('LanguageModel', ModelConfidence.MEDIUM),
        r'transformer\.h\.\d+\.attn': ('GPT-Family', ModelConfidence.MEDIUM),
        r'gpt_neox': ('GPT-NeoX', ModelConfidence.HIGH),
        r'opt\.': ('OPT', ModelConfidence.HIGH),
        r'bloom\.': ('BLOOM', ModelConfidence.HIGH),
        r'roberta': ('RoBERTa', ModelConfidence.HIGH),
        r'bert\.': ('BERT', ModelConfidence.HIGH),
        r'encoder\.layer\.\d+\.attention': ('Transformer-Encoder', ModelConfidence.MEDIUM),
        r'decoder\.layer\.\d+\.self_attention': ('Transformer-Decoder', ModelConfidence.MEDIUM),
        r'palm\.': ('PaLM', ModelConfidence.HIGH),
        r'gemma\.': ('Gemma', ModelConfidence.HIGH),
        r'qwen\.': ('Qwen', ModelConfidence.HIGH),
        r'phi\.': ('Phi', ModelConfidence.HIGH),
        r'mamba\.': ('Mamba', ModelConfidence.HIGH),
        r'falcon\.': ('Falcon', ModelConfidence.HIGH),
        r't5\.': ('T5', ModelConfidence.HIGH),

        # Vision model patterns
        r'vision_model': ('VisionModel', ModelConfidence.HIGH),
        r'encoder\.blocks': ('VisionEncoder', ModelConfidence.MEDIUM),
        r'visual\.transformer': ('CLIPVision', ModelConfidence.HIGH),
        r'vit': ('ViT', ModelConfidence.HIGH),
        r'resnet': ('ResNet', ModelConfidence.HIGH),
        r'efficientnet': ('EfficientNet', ModelConfidence.HIGH),
        r'convnext': ('ConvNeXt', ModelConfidence.HIGH),
        r'dino': ('DINO', ModelConfidence.HIGH),
        r'dinov2': ('DINOv2', ModelConfidence.HIGH),
        r'swin_transformer': ('SwinTransformer', ModelConfidence.HIGH),
        r'maxvit': ('MaxViT', ModelConfidence.HIGH),
        r'beit': ('BEiT', ModelConfidence.HIGH),
        r'perceiver': ('Perceiver', ModelConfidence.HIGH),
        r'mobilenet': ('MobileNet', ModelConfidence.HIGH),
        r'densenet': ('DenseNet', ModelConfidence.HIGH),

        # Multimodal model patterns
        r'clip\.': ('CLIP', ModelConfidence.HIGH),
        r'text_encoder': ('TextEncoder', ModelConfidence.MEDIUM),
        r'image_encoder': ('ImageEncoder', ModelConfidence.MEDIUM),
        r'text_model': ('TextModel', ModelConfidence.MEDIUM),
        r'vision_tower': ('VisionTower', ModelConfidence.HIGH),
        r'blip': ('BLIP', ModelConfidence.HIGH),
        r'flamingo': ('Flamingo', ModelConfidence.HIGH),
        r'image_text_model': ('ImageTextModel', ModelConfidence.HIGH),
        r'siglip': ('SigLIP', ModelConfidence.HIGH),

        # Object detection/segmentation
        r'detr': ('DETR', ModelConfidence.HIGH),
        r'maskformer': ('MaskFormer', ModelConfidence.HIGH),
        r'faster_rcnn': ('FasterRCNN', ModelConfidence.HIGH),
        r'yolo': ('YOLO', ModelConfidence.HIGH),
        r'segformer': ('SegFormer', ModelConfidence.HIGH),
        r'mask_rcnn': ('MaskRCNN', ModelConfidence.HIGH),

        # Audio model patterns
        r'wav2vec': ('Wav2Vec', ModelConfidence.HIGH),
        r'whisper\.encoder': ('WhisperEncoder', ModelConfidence.HIGH),
        r'whisper\.decoder': ('WhisperDecoder', ModelConfidence.HIGH),
        r'audio_encoder': ('AudioEncoder', ModelConfidence.MEDIUM),
        r'hubert': ('HuBERT', ModelConfidence.HIGH),
        r'musicgen': ('MusicGen', ModelConfidence.HIGH),
        r'bark': ('Bark', ModelConfidence.HIGH),
        r'encodec': ('EnCodec', ModelConfidence.HIGH),
        r'audiocraft': ('AudioCraft', ModelConfidence.HIGH),

        # LoRA and adapter patterns
        r'lora_': ('LoRA', ModelConfidence.HIGH),
        r'adapter_model': ('Adapter', ModelConfidence.HIGH),
        r'adapter_blocks': ('Adapter', ModelConfidence.HIGH),
        r'prefix_tuning': ('PrefixTuning', ModelConfidence.HIGH),
        r'prompt_tuning': ('PromptTuning', ModelConfidence.HIGH),

        # LyCORIS patterns
        r'lokr_': ('LyCORIS_LoKR', ModelConfidence.HIGH),
        r'loha_': ('LyCORIS_LoHA', ModelConfidence.HIGH),
        r'locon_': ('LyCORIS_LoCon', ModelConfidence.HIGH),
        r'loha\d+_': ('LyCORIS_LoHA', ModelConfidence.HIGH),

        # Embedding model patterns
        r'token_embedding': ('TokenEmbedding', ModelConfidence.HIGH),
        r'embedding_layer': ('EmbeddingModel', ModelConfidence.MEDIUM),
        r'word_embeddings': ('WordEmbedding', ModelConfidence.HIGH),
        r'sentence_transformer': ('SentenceTransformer', ModelConfidence.HIGH),

        # Segment Anything Model
        r'sam\.': ('SAM', ModelConfidence.HIGH),
        r'mask_decoder': ('MaskDecoder', ModelConfidence.MEDIUM),

        # RetinaFace/FaceDetection
        r'retinaface': ('RetinaFace', ModelConfidence.HIGH),
        r'facenet': ('FaceNet', ModelConfidence.HIGH),
        r'arcface': ('ArcFace', ModelConfidence.HIGH),
        r'insightface': ('InsightFace', ModelConfidence.HIGH),

        # GANs and other generative models (non-diffusion)
        r'generator\.': ('Generator', ModelConfidence.MEDIUM),
        r'discriminator\.': ('Discriminator', ModelConfidence.MEDIUM),
        r'stylegan': ('StyleGAN', ModelConfidence.HIGH),
        r'pgan': ('ProgressiveGAN', ModelConfidence.HIGH),
        r'biggan': ('BigGAN', ModelConfidence.HIGH),

        # Reinforcement Learning models
        r'policy_net': ('PolicyNetwork', ModelConfidence.HIGH),
        r'value_net': ('ValueNetwork', ModelConfidence.HIGH),
        r'dqn': ('DQN', ModelConfidence.HIGH),
        r'ppo': ('PPO', ModelConfidence.HIGH),
        r'actor_critic': ('ActorCritic', ModelConfidence.HIGH),

        # Graph Neural Networks
        r'gnn\.': ('GraphNeuralNetwork', ModelConfidence.HIGH),
        r'gcn\.': ('GraphConvNetwork', ModelConfidence.HIGH),
        r'gat\.': ('GraphAttentionNetwork', ModelConfidence.HIGH),

        # Recommendation Systems
        r'recommender': ('RecommenderSystem', ModelConfidence.HIGH),
        r'matrix_fact': ('MatrixFactorization', ModelConfidence.HIGH),
        r'collaborative_filter': ('CollaborativeFiltering', ModelConfidence.HIGH),

        # Speech synthesis
        r'tacotron': ('Tacotron', ModelConfidence.HIGH),
        r'wavenet': ('WaveNet', ModelConfidence.HIGH),
        r'fastspeech': ('FastSpeech', ModelConfidence.HIGH),
        r'vits': ('VITS', ModelConfidence.HIGH),
        r'hifigan': ('HiFiGAN', ModelConfidence.HIGH),

        # Time Series models
        r'lstm\.': ('LSTM', ModelConfidence.HIGH),
        r'gru\.': ('GRU', ModelConfidence.HIGH),
        r'rnn\.': ('RNN', ModelConfidence.HIGH),
        r'tcn\.': ('TemporalConvNet', ModelConfidence.HIGH),
        r'transformer_ts': ('TimeSeriesTransformer', ModelConfidence.HIGH),

        # Quantized models
        r'quantized': ('QuantizedModel', ModelConfidence.MEDIUM),
        r'qat\.': ('QuantizationAwareTraining', ModelConfidence.HIGH),
        r'quant_layers': ('QuantizedLayers', ModelConfidence.HIGH),
    }

    # Dimension size hints for model types
    DIMENSION_HINTS = {
        # Common embedding dimensions for language models
        768: ('BERT-Base/ViT-Base', ModelConfidence.MEDIUM),
        1024: ('BERT-Large/ResNet', ModelConfidence.MEDIUM),
        1280: ('CLIP-ViT-Large', ModelConfidence.MEDIUM),
        2048: ('CLIP-ViT-bigG', ModelConfidence.MEDIUM),
        2560: ('MPT-7B', ModelConfidence.MEDIUM),
        3072: ('BERT-XXLarge', ModelConfidence.MEDIUM),
        4096: ('LLaMA-7B', ModelConfidence.MEDIUM),
        5120: ('LLaMA-13B', ModelConfidence.MEDIUM),
        6656: ('LLaMA2-70B', ModelConfidence.MEDIUM),
        8192: ('GPT-J-6B', ModelConfidence.MEDIUM),
        # Common CNN channel dimensions
        64: ('CNN-Small', ModelConfidence.LOW),
        128: ('CNN-Medium', ModelConfidence.LOW),
        256: ('CNN-Large', ModelConfidence.LOW),
        512: ('ResNet-Base', ModelConfidence.LOW),
    }

    # Known networks and their string identifiers in metadata
    KNOWN_NETWORKS = {
        'networks.lora': 'LoRA',
        'lycoris.kohya': 'LyCORIS',
        'locon': 'LoCon',
        'loha': 'LoHA',
        'lokr': 'LoKR',
        'ia3': 'IA3',
        'adapter': 'Adapter',
        'prompt_tuning': 'PromptTuning',
        'prefix_tuning': 'PrefixTuning',
        'p_tuning': 'P-Tuning',
        'bitfit': 'BitFit',
        'compacter': 'Compacter',
    }

    def __init__(self):
        """Initialize the safetensors analyzer."""
        super().__init__()

    def get_supported_extensions(self) -> set:
        """
        Get the file extensions supported by this analyzer.

        Returns:
            Set of supported file extensions
        """
        return {'.safetensors'}

    def analyze(self, file_path: str) -> Tuple[str, ModelConfidence, Dict[str, Any]]:
        """
        Analyze a safetensors file to determine its model type and metadata.

        Args:
            file_path: Path to the safetensors file

        Returns:
            Tuple of (model_type, confidence, metadata)

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid safetensors file
            Exception: If there is an error during analysis
        """
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read safetensors header
            metadata, tensor_info = self._read_safetensors_metadata(file_path)

            # Extract model type
            model_type, confidence = self._determine_model_type(metadata, tensor_info)

            # Process and clean the metadata
            cleaned_metadata = self._process_metadata(metadata)

            # Add information about the tensors
            if tensor_info:
                tensor_metadata = self._extract_tensor_metadata(tensor_info)
                cleaned_metadata.update(tensor_metadata)

            return model_type, confidence, cleaned_metadata

        except Exception as e:
            logging.error(f"Error analyzing safetensors file {file_path}: {e}")
            raise

    def _read_safetensors_metadata(self, file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Read metadata from a safetensors file without loading tensors.

        Args:
            file_path: Path to the safetensors file

        Returns:
            Tuple of (metadata dict, tensor info dict)
        """
        try:
            # Try to use safetensors library if available
            from safetensors import safe_open
            with safe_open(file_path, framework="pt", device="meta") as f:
                # Extract metadata
                metadata = {}
                if hasattr(f, 'metadata') and f.metadata is not None:
                    metadata = dict(f.metadata())

                # Extract tensor info
                tensor_info = {}
                for key in f.keys():
                    info = f.get_tensor_info(key)
                    tensor_info[key] = {
                        'dtype': str(info.dtype),
                        'shape': info.shape,
                        'data_offsets': (info.data_offsets[0], info.data_offsets[1])
                    }

                return metadata, tensor_info

        except ImportError:
            # Fallback to manual parsing if safetensors not available
            return self._manual_parse_safetensors(file_path)

    def _manual_parse_safetensors(self, file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Manually parse a safetensors file to extract metadata and tensor info.

        Args:
            file_path: Path to the safetensors file

        Returns:
            Tuple of (metadata dict, tensor info dict)
        """
        with open(file_path, 'rb') as f:
            # Read header length (8 bytes)
            header_length_bytes = f.read(8)
            if len(header_length_bytes) != 8:
                raise ValueError("Invalid safetensors file: can't read header length")

            header_length = int.from_bytes(header_length_bytes, byteorder='little')

            # Read the header (JSON)
            header_bytes = f.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError("Invalid safetensors file: header length mismatch")

            header = json.loads(header_bytes)

            # Extract metadata (in "__metadata__" field if present)
            metadata = header.get("__metadata__", {})

            # Rest of the header contains tensor info
            tensor_info = {k: v for k, v in header.items() if k != "__metadata__"}

            return metadata, tensor_info

    def _determine_model_type(
            self,
            metadata: Dict[str, Any],
            tensor_info: Dict[str, Any]
    ) -> Tuple[str, ModelConfidence]:
        """
        Determine model type from metadata and tensor information.

        Args:
            metadata: Metadata dictionary from the safetensors file
            tensor_info: Information about tensors in the file

        Returns:
            Tuple of (model_type, confidence)
        """
        # Check from most specific to least specific sources

        # 1. First check for explicit type in metadata
        if metadata.get('model_type'):
            return metadata['model_type'], ModelConfidence.HIGH

        # 2. Check for huggingface model info
        if metadata.get('_name_or_path'):
            model_path = metadata['_name_or_path']
            # Extract model family from the path
            if 'bert' in model_path.lower():
                return 'BERT', ModelConfidence.HIGH
            if 'gpt' in model_path.lower():
                return 'GPT', ModelConfidence.HIGH
            if 't5' in model_path.lower():
                return 'T5', ModelConfidence.HIGH
            if 'llama' in model_path.lower():
                return 'LLaMA', ModelConfidence.HIGH
            if 'clip' in model_path.lower():
                return 'CLIP', ModelConfidence.HIGH
            if 'resnet' in model_path.lower():
                return 'ResNet', ModelConfidence.HIGH
            if 'vit' in model_path.lower():
                return 'ViT', ModelConfidence.HIGH
            if 'wav2vec' in model_path.lower():
                return 'Wav2Vec', ModelConfidence.HIGH
            if 'whisper' in model_path.lower():
                return 'Whisper', ModelConfidence.HIGH

            # Return the model name itself as type with medium confidence
            return model_path, ModelConfidence.MEDIUM

        # 3. Check for specific network types commonly found in metadata
        for net_key, net_field in [
            ('ss_network_type', None),
            ('ss_network_module', None),
            ('network_type', None),
            ('network_module', None)
        ]:
            if net_key in metadata:
                net_value = metadata[net_key]
                # Look up the network type in our known networks
                for known_key, known_type in self.KNOWN_NETWORKS.items():
                    if known_key in net_value.lower():
                        return known_type, ModelConfidence.HIGH

                # If not in our known types but has a value, use that
                return f"Network_{net_value}", ModelConfidence.MEDIUM

        # 4. Check in model configuration if present
        if 'modelspec.config' in metadata:
            config = metadata['modelspec.config']
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except json.JSONDecodeError:
                    pass

            if isinstance(config, dict):
                if 'model_type' in config:
                    return config['model_type'], ModelConfidence.HIGH
                if 'architectures' in config and config['architectures']:
                    return config['architectures'][0], ModelConfidence.HIGH

        # 5. Exclude diffusion models explicitly
        diffusion_indicators = [
            'unet', 'vae', 'stable_diffusion', 'diffusion_model',
            'latent_diffusion', 'ldm', 'clip_l', 'controlnet'
        ]
        tensor_keys = ' '.join(tensor_info.keys()).lower()
        if any(indicator in tensor_keys for indicator in diffusion_indicators):
            return 'UnsupportedDiffusionModel', ModelConfidence.HIGH

        # 6. Check tensor keys against known patterns
        for key in tensor_info.keys():
            for pattern, (model_type, confidence) in self.MODEL_PATTERNS.items():
                if re.search(pattern, key):
                    return model_type, confidence

        # 7. If we didn't find a match from patterns, look at dimensions
        for key, info in tensor_info.items():
            if 'shape' in info:
                shape = info['shape']
                for dim in shape:
                    if dim in self.DIMENSION_HINTS:
                        model_type, confidence = self.DIMENSION_HINTS[dim]
                        return model_type, confidence

        # 8. If still no determination, use heuristics from available info
        if 'ss_training_comment' in metadata or 'training_comment' in metadata:
            comment_key = 'ss_training_comment' if 'ss_training_comment' in metadata else 'training_comment'
            comment = metadata[comment_key].lower()

            if 'lora' in comment:
                return 'LoRA', ModelConfidence.MEDIUM
            if 'bert' in comment:
                return 'BERT', ModelConfidence.MEDIUM
            if 'clip' in comment:
                return 'CLIP', ModelConfidence.MEDIUM
            if 'whisper' in comment:
                return 'Whisper', ModelConfidence.MEDIUM
            if 'vision' in comment:
                return 'VisionModel', ModelConfidence.MEDIUM
            if 'gan' in comment:
                return 'GAN', ModelConfidence.MEDIUM

        # 9. Check if it might be a specialized model architecture
        # Check for RNN-based architectures
        if any('lstm' in k.lower() or 'gru' in k.lower() or 'rnn' in k.lower() for k in tensor_info):
            return 'RecurrentNN', ModelConfidence.MEDIUM

        # Check for graph neural networks
        if any('graph' in k.lower() or 'gcn' in k.lower() or 'gat' in k.lower() for k in tensor_info):
            return 'GraphNN', ModelConfidence.MEDIUM

        # Check for CNN architectures
        if any('conv' in k.lower() for k in tensor_info) and not any('transformer' in k.lower() for k in tensor_info):
            return 'ConvolutionalNN', ModelConfidence.MEDIUM

        # 10. Analyze tensor count and key patterns
        if len(tensor_info) < 10:
            if any('embedding' in k.lower() for k in tensor_info):
                return 'Embedding', ModelConfidence.LOW
            if any('lora' in k.lower() for k in tensor_info):
                return 'LoRA', ModelConfidence.LOW
            return 'SmallModel', ModelConfidence.LOW
        elif len(tensor_info) < 100:
            if any('attention' in k.lower() for k in tensor_info):
                return 'TransformerComponent', ModelConfidence.LOW
            return 'MediumModel', ModelConfidence.LOW
        else:
            return 'LargeModel', ModelConfidence.LOW

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and clean metadata.

        Args:
            metadata: Raw metadata from the safetensors file

        Returns:
            Cleaned and processed metadata
        """
        result = {}

        # Copy all metadata fields
        for key, value in metadata.items():
            # Try to decode JSON strings
            if key in self.JSON_FIELDS and isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
            else:
                result[key] = value

        # Some common conversions for consistency

        # Extract creator information
        for creator_key in ['ss_creator', 'creator', 'author']:
            if creator_key in result:
                result['creator'] = result[creator_key]
                break

        # Extract description
        for desc_key in ['ss_training_comment', 'description', 'ss_tag_frequency', 'comment']:
            if desc_key in result and not result.get('description'):
                result['description'] = result[desc_key]

        # Extract tags
        for tag_key in ['ss_tag_frequency', 'tags', 'ss_tags']:
            if tag_key in result:
                try:
                    if isinstance(result[tag_key], str):
                        tag_data = json.loads(result[tag_key])
                    else:
                        tag_data = result[tag_key]

                    if isinstance(tag_data, dict):
                        # Tag frequency format
                        result['tags'] = list(tag_data.keys())
                    elif isinstance(tag_data, list):
                        # Regular tag list
                        result['tags'] = tag_data
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Extract training parameters
        training_params = {}

        # Check for common training parameters
        param_mapping = {
            'ss_learning_rate': 'learning_rate',
            'learning_rate': 'learning_rate',
            'ss_epoch': 'epochs',
            'epoch': 'epochs',
            'epochs': 'epochs',
            'ss_steps': 'steps',
            'steps': 'steps',
            'ss_network_dim': 'network_dim',
            'network_dim': 'network_dim',
            'ss_network_alpha': 'network_alpha',
            'network_alpha': 'network_alpha',
            'ss_batch_size': 'batch_size',
            'batch_size': 'batch_size',
            'optimizer': 'optimizer',
            'scheduler': 'scheduler',
            'weight_decay': 'weight_decay',
            'warmup_steps': 'warmup_steps'
        }

        for source_key, target_key in param_mapping.items():
            if source_key in result:
                training_params[target_key] = result[source_key]

        if training_params:
            result['training_params'] = training_params

        # Look for config information
        if 'modelspec.config' in result and isinstance(result['modelspec.config'], dict):
            result['config'] = result['modelspec.config']

        # Try to extract version information
        for version_key in ['ss_sd_model_name', 'model_version', 'version']:
            if version_key in result:
                result['version'] = result[version_key]
                break

        return result

    def _extract_tensor_metadata(self, tensor_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from tensor information.

        Args:
            tensor_info: Information about tensors in the file

        Returns:
            Metadata extracted from tensors
        """
        result = {
            'tensor_count': len(tensor_info),
            'tensor_shapes': {},
            'tensor_dtypes': {}
        }

        # Count occurrences of each shape and dtype
        shapes = Counter()
        dtypes = Counter()

        # Track total parameters
        total_params = 0

        for key, info in tensor_info.items():
            if 'shape' in info:
                shape = tuple(info['shape'])
                shapes[shape] += 1

                # Calculate number of parameters in this tensor
                param_count = 1
                for dim in shape:
                    param_count *= dim
                total_params += param_count

            if 'dtype' in info:
                dtype = info['dtype']
                dtypes[dtype] += 1

        # Store the most common shapes and dtypes
        result['common_shapes'] = dict(shapes.most_common(5))
        result['common_dtypes'] = dict(dtypes.most_common(5))
        result['total_parameters'] = total_params

        # Format the total parameters in a readable way
        if total_params > 1_000_000_000:
            result['parameter_count_formatted'] = f"{total_params / 1_000_000_000:.2f}B"
        elif total_params > 1_000_000:
            result['parameter_count_formatted'] = f"{total_params / 1_000_000:.2f}M"
        elif total_params > 1_000:
            result['parameter_count_formatted'] = f"{total_params / 1_000:.2f}K"

        # Try to determine model dimensions
        model_dims = self._extract_model_dims(tensor_info)
        if model_dims:
            result['model_dimensions'] = model_dims

        # Analyze architecture by looking at tensor key patterns
        architecture_features = self._analyze_architecture(tensor_info)
        if architecture_features:
            result['architecture_features'] = architecture_features

        return result

    def _extract_model_dims(self, tensor_info: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """
        Try to extract model dimensions from tensor shapes.

        Args:
            tensor_info: Information about tensors in the file

        Returns:
            Dictionary of model dimensions or None if can't determine
        """
        dims = {}

        # Check for embedding dimensions
        embed_keys = [k for k in tensor_info if any(p in k.lower() for p in
                                                    ['embed', 'token', 'word'])]
        for key in embed_keys:
            if 'shape' in tensor_info[key]:
                shape = tensor_info[key]['shape']
                if len(shape) == 2:
                    # Usually the second dimension is the embedding dimension
                    dims['embedding_dim'] = shape[1]
                    # And the first dimension might be vocabulary size
                    if shape[0] > 100:  # Only if reasonably sized for a vocab
                        dims['vocab_size'] = shape[0]
                    break

        # Check for hidden dimensions in transformer models
        hidden_keys = [k for k in tensor_info if any(p in k.lower() for p in
                                                     ['mlp', 'attention', 'linear', 'dense', 'fc'])]
        for key in hidden_keys:
            if 'shape' in tensor_info[key]:
                shape = tensor_info[key]['shape']
                if len(shape) == 2 and shape[0] > 100 and shape[1] > 100:
                    dims['hidden_dim'] = max(shape)
                    break

        # Check for number of layers by looking at repeated patterns
        layer_patterns = [
            r'layers\.(\d+)\.',
            r'layer\.(\d+)\.',
            r'h\.(\d+)\.',
            r'blocks\.(\d+)\.'
        ]

        max_layer = -1
        for pattern in layer_patterns:
            for key in tensor_info:
                matches = re.findall(pattern, key)
                if matches:
                    layer_num = max(int(m) for m in matches)
                    max_layer = max(max_layer, layer_num)

        if max_layer >= 0:
            dims['num_layers'] = max_layer + 1  # +1 because zero-indexed

        # Check for attention heads by looking for common head dimensions
        qkv_keys = [k for k in tensor_info if any(p in k.lower() for p in
                                                  ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'attention'])]

        for key in qkv_keys:
            if 'shape' in tensor_info[key]:
                shape = tensor_info[key]['shape']
                # Many models have head dimension as a factor of the hidden dimension
                if len(shape) == 2 and 'hidden_dim' in dims and dims['hidden_dim'] % shape[1] == 0:
                    potential_num_heads = dims['hidden_dim'] // shape[1]
                    # Common head counts are in range of 8-128
                    if 8 <= potential_num_heads <= 128:
                        dims['num_attention_heads'] = potential_num_heads
                        dims['head_dim'] = shape[1]
                        break

        # Get input dimensions for vision models
        vision_keys = [k for k in tensor_info if any(p in k.lower() for p in
                                                     ['patch_embed', 'conv1', 'stem', 'first_conv'])]
        for key in vision_keys:
            if 'shape' in tensor_info[key]:
                shape = tensor_info[key]['shape']
                if len(shape) == 4:  # Conv layers typically have 4D shapes
                    dims['input_channels'] = shape[1]
                    break

        # Get dimensions for RNN models
        rnn_keys = [k for k in tensor_info if any(p in k.lower() for p in
                                                  ['lstm', 'gru', 'rnn'])]
        for key in rnn_keys:
            if 'shape' in tensor_info[key] and 'hidden_dim' not in dims:
                shape = tensor_info[key]['shape']
                if len(shape) >= 2:
                    # For LSTM/GRU models, hidden dimension is often in these tensors
                    dims['hidden_dim'] = shape[-1]
                    break

        return dims if dims else None

    def _analyze_architecture(self, tensor_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze tensor keys to determine architectural features.

        Args:
            tensor_info: Information about tensors in the file

        Returns:
            Dictionary of architectural features
        """
        features = {}

        # Check for common architectural components
        components = {
            'attention': ['attention', 'self_attn', 'mha'],
            'mlp': ['mlp', 'feed_forward', 'ffn'],
            'normalization': ['layernorm', 'rmsnorm', 'norm'],
            'embedding': ['embed', 'wte', 'token_emb'],
            'positional_encoding': ['pos_emb', 'position', 'rotary'],
            'convolutional': ['conv', 'convolution'],
            'recurrent': ['lstm', 'gru', 'rnn'],
            'pooling': ['pool', 'max_pool', 'avg_pool'],
            'residual': ['residual', 'skip'],
            'batch_norm': ['bn', 'batch_norm'],
        }

        found_components = {}
        for component, patterns in components.items():
            for key in tensor_info:
                if any(p in key.lower() for p in patterns):
                    found_components[component] = True
                    break

        if found_components:
            features['components'] = list(found_components.keys())

        # Check for specific attention types
        attention_types = {
            'self_attention': ['self_attn', 'self_attention'],
            'cross_attention': ['cross_attn', 'cross_attention', 'encoder_decoder_attention'],
            'multi_head': ['multihead', 'multi_head', 'mha'],
            'global_attention': ['global_attn', 'global_attention'],
            'local_attention': ['local_attn', 'local_attention'],
            'grouped_attention': ['grouped_attn', 'grouped_query'],
            'sliding_window': ['sliding_window', 'window_attn'],
        }

        found_attention_types = {}
        for att_type, patterns in attention_types.items():
            for key in tensor_info:
                if any(p in key.lower() for p in patterns):
                    found_attention_types[att_type] = True
                    break

        if found_attention_types:
            features['attention_types'] = list(found_attention_types.keys())

        # Check for activation functions
        activations = {
            'gelu': ['gelu'],
            'swiglu': ['swiglu', 'silu'],
            'relu': ['relu'],
            'silu': ['silu'],
            'sigmoid': ['sigmoid'],
            'tanh': ['tanh'],
            'leaky_relu': ['leaky_relu'],
            'mish': ['mish'],
            'softmax': ['softmax'],
        }

        found_activations = {}
        for act, patterns in activations.items():
            for key in tensor_info:
                if any(p in key.lower() for p in patterns):
                    found_activations[act] = True
                    break

        if found_activations:
            features['activations'] = list(found_activations.keys())

        # Support for encoder-decoder architectures:
        if any('encoder' in k.lower() for k in tensor_info) and any('decoder' in k.lower() for k in tensor_info):
            features['architecture_type'] = 'encoder_decoder'
        elif any('encoder' in k.lower() for k in tensor_info):
            features['architecture_type'] = 'encoder_only'
        elif any('decoder' in k.lower() for k in tensor_info):
            features['architecture_type'] = 'decoder_only'
        elif any(x in k.lower() for k in tensor_info for x in ['transformer', 'attention']):
            features['architecture_type'] = 'transformer'
        elif any(x in k.lower() for k in tensor_info for x in ['conv', 'resnet', 'vgg']):
            features['architecture_type'] = 'cnn'
        elif any(x in k.lower() for k in tensor_info for x in ['lstm', 'gru', 'rnn']):
            features['architecture_type'] = 'rnn'
        elif any(x in k.lower() for k in tensor_info for x in ['gcn', 'gat', 'graph']):
            features['architecture_type'] = 'graph_neural_network'

        # Check for quantization hints
        if any(any(q in k.lower() for q in ['quant', 'qweight', 'qproj', 'int8', 'int4', 'fp8'])
               for k in tensor_info):
            features['quantized'] = True

            # Try to determine quantization type
            if any('int8' in k.lower() for k in tensor_info):
                features['quantization_type'] = 'int8'
            elif any('int4' in k.lower() for k in tensor_info):
                features['quantization_type'] = 'int4'
            elif any('fp8' in k.lower() for k in tensor_info):
                features['quantization_type'] = 'fp8'
            elif any('fp16' in k.lower() for k in tensor_info):
                features['quantization_type'] = 'fp16'

        # Check if model has sparse architecture
        if any(any(s in k.lower() for s in ['sparse', 'sparsity', 'mixtral'])
               for k in tensor_info):
            features['sparse_architecture'] = True

        # Check for specific model architecture indicators
        if any('soft_prompt' in k.lower() for k in tensor_info):
            features['soft_prompt_tuning'] = True

        if any('bilinear' in k.lower() for k in tensor_info):
            features['bilinear_layers'] = True

        if any('gate' in k.lower() for k in tensor_info):
            features['gated_mechanism'] = True

        if any('expert' in k.lower() for k in tensor_info):
            features['mixture_of_experts'] = True

        return features
