"""
Example showing how to analyze Stable Diffusion and other Diffusers models.

This script demonstrates how to use ModelInspector to analyze Stable Diffusion models,
including detecting LoRAs, embeddings, and core model components in different formats
including safetensors, GGUF, and binary formats like .pt, .pth, and .ckpt.
"""
import os
import sys
import re
from pathlib import Path

# Add the parent directory to the path so we can import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_inspector import ModelInspector, SafetyLevel, InspectorConfig
from model_inspector.analyzers import register_analyzer
from model_inspector.analyzers.diffusers import DiffusersRouter
from model_inspector.analyzers.diffusers_bin import DiffusionBinAnalyzer


def format_value(value, max_length=100):
    """Format a value for display, truncating if too long."""
    if isinstance(value, (list, tuple)) and len(value) > 10:
        return f"[{len(value)} items - list truncated]"
    elif isinstance(value, dict) and len(value) > 5:
        return f"{{{len(value)} key-value pairs - dict truncated}}"

    str_value = str(value)
    if len(str_value) > max_length:
        return f"{str_value[:max_length]}... [truncated]"
    return str_value


def main(directory=None):
    """Run the Stable Diffusion/Diffusers example."""
    # Use current directory if none provided
    if directory is None:
        directory = os.getcwd()

    print(f"Analyzing Diffusers models in: {directory}\n")
    print("This example demonstrates analyzing Stable Diffusion models, LoRAs, and embeddings")
    print("Supported formats: .safetensors, .gguf, .pt, .pth, .ckpt (binary formats analyzed safely)\n")

    # Create a configuration that overrides the default safety checks
    # This allows our DiffusionBinAnalyzer to handle the unsafe formats safely
    # Note: All analysis is done safely - no code loading or execution
    config = InspectorConfig(
        recursive=True,
        safety_level=SafetyLevel.UNSAFE,  # Override default safety to let our analyzer run
        show_progress=True
    )

    # Explain what we're doing
    print("Note: All models will be analyzed safely.")
    print("Binary formats (.pt, .pth, .ckpt) will be analyzed without loading model weights.\n")

    # Create the inspector
    inspector = ModelInspector(directory, config=config)

    # Register our analyzers for different formats
    # For safe formats, register the router
    register_analyzer(['.safetensors', '.gguf'], DiffusersRouter)

    # For binary formats, register our safe binary analyzer
    register_analyzer(['.pt', '.pth', '.ckpt', '.checkpoint'], DiffusionBinAnalyzer)

    # List all supported model files
    print("\nDiffusion model files found:")
    all_files = [f for f in inspector.directory_files(show_progress=True)
                if f.lower().endswith(('.safetensors', '.gguf', '.pt', '.pth', '.ckpt', '.checkpoint'))]

    for file_path in all_files:
        print(f"  - {Path(file_path).name}")

    # Analyze all models
    print("\nAnalyzing models...")

    # Analyze models and filter for supported formats
    results = []
    for result in inspector.analyze_directory(show_progress=True):
        if result.filename.lower().endswith(('.safetensors', '.gguf', '.pt', '.pth', '.ckpt', '.checkpoint')):
            results.append(result)

    # Display results in a more organized way
    print(f"\nFound {len(results)} models:")
    for model_info in results:
        print(f"\n{'='*50}")
        print(f"MODEL: {model_info.filename}")
        print(f"{'='*50}")
        print(f"🔍 Type:       {model_info.model_type}")
        print(f"🔒 Confidence: {model_info.confidence.name}")

        # Display trigger words right after confidence
        filename = Path(model_info.filename).stem

        # Check for explicitly set trigger_words first
        if 'trigger_words' in model_info.metadata and model_info.metadata['trigger_words']:
            trigger_words = model_info.metadata['trigger_words']
            if len(trigger_words) == 1:
                print(f"🔑 Trigger:     {trigger_words[0]}")
            else:
                print(f"🔑 Triggers:    {', '.join(trigger_words[:5])}")
                if len(trigger_words) > 5:
                    print(f"               + {len(trigger_words)-5} more")

        # Check for token field in embeddings
        elif 'token' in model_info.metadata:
            print(f"🔑 Trigger:     {model_info.metadata['token']}")

        # Extract from filename as fallback for LoRAs and embeddings
        elif any(x in model_info.model_type for x in ['LoRA', 'Embedding', 'TextualEmbedding']):
            # For embeddings, make it look like a token
            if any(x in model_info.model_type for x in ['Embedding', 'TextualEmbedding']):
                clean_name = re.sub(r'^(emb_|ti_|embedding_|ZM_)', '', filename)
                if not clean_name.startswith('<'):
                    trigger = f"<{clean_name}>"
                else:
                    trigger = clean_name
            # For LoRAs, just clean the filename
            else:
                trigger = filename.replace('_', ' ')

            print(f"🔑 Trigger:     {trigger}")

        print(f"💾 Size:       {model_info.file_size:,} bytes")
        print(f"📄 Format:     {model_info.metadata.get('format', 'Unknown')}")

        # Display note for binary formats
        if model_info.filename.lower().endswith(('.pt', '.pth', '.ckpt', '.checkpoint')):
            print(f"ℹ️  Analysis:    Binary format analyzed via safe inspection only")

        # More targeted metadata display based on model type
        if not model_info.metadata:
            continue

        print("\n📋 Key Information:")

        # For binary analyzed files (PT, PTH, CKPT)
        if model_info.filename.lower().endswith(('.pt', '.pth', '.ckpt', '.checkpoint')):
            if 'safe_analysis_only' in model_info.metadata:
                print(f"  • Safe analysis:     {model_info.metadata.get('safe_analysis_only')}")
            if 'has_config_file' in model_info.metadata and model_info.metadata['has_config_file']:
                print(f"  • Config file:       {model_info.metadata.get('config_path', 'Yes')}")
            if 'sd_version' in model_info.metadata:
                print(f"  • SD Version:        {model_info.metadata.get('sd_version')}")
            if 'model_family' in model_info.metadata:
                print(f"  • Model family:      {model_info.metadata.get('model_family')}")
            if 'component_type' in model_info.metadata:
                print(f"  • Component:         {model_info.metadata.get('component_type')}")
            if 'diffusion_marker_detected' in model_info.metadata:
                print(f"  • Detected marker:   {model_info.metadata.get('diffusion_marker', 'Yes')}")

        # For LoRA models, show their attributes
        elif 'LoRA' in model_info.model_type:
            if 'network_dim' in model_info.metadata:
                print(f"  • Rank/Dimension:    {model_info.metadata.get('network_dim')}")
            if 'network_alpha' in model_info.metadata:
                print(f"  • Alpha:             {model_info.metadata.get('network_alpha')}")
            if 'model_compatibility' in model_info.metadata:
                print(f"  • Compatible with:   {model_info.metadata.get('model_compatibility')}")
            if 'ss_output_name' in model_info.metadata:
                print(f"  • Name:              {model_info.metadata.get('ss_output_name')}")
            # Show trigger words again in the attributes section for clarity
            if 'trigger_words' in model_info.metadata and model_info.metadata['trigger_words']:
                trigger_words = model_info.metadata['trigger_words']
                print(f"  • Trigger word(s):   {format_value(trigger_words)}")

        # For embeddings, show their attributes
        elif 'Embedding' in model_info.model_type or 'TextualEmbedding' in model_info.model_type:
            if 'token' in model_info.metadata:
                print(f"  • Token:             {model_info.metadata.get('token')}")
            if 'vector_count' in model_info.metadata:
                print(f"  • Vector count:      {model_info.metadata.get('vector_count')}")
            if 'embedding_dim' in model_info.metadata:
                print(f"  • Embedding dim:     {model_info.metadata.get('embedding_dim')}")
            if 'model_compatibility' in model_info.metadata:
                print(f"  • Compatible with:   {model_info.metadata.get('model_compatibility')}")
            # Show trigger words again in the attributes section
            if 'trigger_words' in model_info.metadata and model_info.metadata['trigger_words']:
                trigger_words = model_info.metadata['trigger_words']
                print(f"  • Trigger word(s):   {format_value(trigger_words)}")

        # For UNet, VAE, TextEncoder, show specific model attributes
        elif any(x in model_info.model_type for x in ['UNet', 'VAE', 'TextEncoder', 'Diffusers']):
            if 'architectures' in model_info.metadata:
                print(f"  • Architecture:      {format_value(model_info.metadata.get('architectures'))}")
            if 'cross_attention_dim' in model_info.metadata:
                print(f"  • Cross-attn dim:    {model_info.metadata.get('cross_attention_dim')}")
            if 'in_channels' in model_info.metadata:
                print(f"  • Input channels:    {model_info.metadata.get('in_channels')}")
            if 'out_channels' in model_info.metadata:
                print(f"  • Output channels:   {model_info.metadata.get('out_channels')}")
            if 'sample_size' in model_info.metadata:
                print(f"  • Sample size:       {model_info.metadata.get('sample_size')}")

        # For ControlNets/adapters
        elif 'ControlNet' in model_info.model_type or 'Adapter' in model_info.model_type:
            if 'architectures' in model_info.metadata:
                print(f"  • Architecture:      {format_value(model_info.metadata.get('architectures'))}")
            if 'controlnet_conditioning_channel_order' in model_info.metadata:
                print(f"  • Channel order:     {model_info.metadata.get('controlnet_conditioning_channel_order')}")
            if 'conditioning_embedding_out_channels' in model_info.metadata:
                print(f"  • Cond. channels:    {model_info.metadata.get('conditioning_embedding_out_channels')}")

        # GGUF-specific attributes
        if 'format' in model_info.metadata and model_info.metadata['format'] == 'gguf':
            if 'format_version' in model_info.metadata:
                print(f"  • GGUF Version:      {model_info.metadata.get('format_version')}")
            if 'general_params' in model_info.metadata:
                print(f"  • General params:    {format_value(model_info.metadata.get('general_params'))}")
            if 'is_diffusion_model' in model_info.metadata:
                print(f"  • Diffusion model:   Yes")
            if 'has_unet' in model_info.metadata and model_info.metadata['has_unet']:
                print(f"  • Contains UNet:     Yes")
            if 'has_vae' in model_info.metadata and model_info.metadata['has_vae']:
                print(f"  • Contains VAE:      Yes")
            if 'has_text_encoder' in model_info.metadata and model_info.metadata['has_text_encoder']:
                print(f"  • Contains Text Enc: Yes")

        # Show tensor info for all models
        if 'tensor_count' in model_info.metadata:
            print(f"\n🧮 Tensor Information:")
            print(f"  • Tensor count:       {model_info.metadata.get('tensor_count')}")
            if 'common_shapes' in model_info.metadata:
                shapes = model_info.metadata.get('common_shapes')
                if shapes:
                    print(f"  • Most common shapes:  {format_value(list(shapes.items())[:3])}")
            if 'common_dtypes' in model_info.metadata:
                print(f"  • Data types:         {format_value(model_info.metadata.get('common_dtypes'))}")

        # For binary formats, show additional evidence found
        if model_info.filename.lower().endswith(('.pt', '.pth', '.ckpt', '.checkpoint')):
            if 'evidence_summary' in model_info.metadata:
                print(f"\n🔍 Evidence Found:")
                for category, items in model_info.metadata.get('evidence_summary', {}).items():
                    print(f"  • {category.title()}: {format_value(items)}")


if __name__ == "__main__":
    # Allow specifying a directory as a command-line argument
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    main(directory)
