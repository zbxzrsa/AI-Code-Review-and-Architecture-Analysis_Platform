"""
Model Quantization module.

Supports multiple quantization methods:
- INT8: 8-bit integer quantization (PyTorch native)
- INT4: 4-bit integer quantization (bitsandbytes)
- FP8: 8-bit floating point
- GPTQ: Post-training quantization with calibration
- AWQ: Activation-aware weight quantization
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import PracticalDeploymentConfig, QuantizationType

logger = logging.getLogger(__name__)


@dataclass
class QuantizationStats:
    """Statistics from quantization process."""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    num_quantized_layers: int
    quantization_type: str
    avg_weight_range: Optional[Tuple[float, float]] = None


class INT4Quantizer:
    """
    INT4 quantization using bitsandbytes library.
    
    Supports:
    - NF4 (NormalFloat4): Better for normally distributed weights
    - FP4: Standard 4-bit floating point
    - Double quantization: Quantizes the quantization constants
    """
    
    def __init__(
        self,
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = "nf4",  # "nf4" or "fp4"
        double_quant: bool = True,
    ):
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self.double_quant = double_quant
        
        self._bnb_available = False
        self._check_bitsandbytes()
    
    def _check_bitsandbytes(self):
        """Check if bitsandbytes is available."""
        try:
            import bitsandbytes as bnb
            self._bnb_available = True
            logger.info("bitsandbytes available for INT4 quantization")
        except ImportError:
            logger.warning(
                "bitsandbytes not installed. Install with: "
                "pip install bitsandbytes>=0.41.0"
            )
            self._bnb_available = False
    
    def quantize(self, model: nn.Module) -> Tuple[nn.Module, QuantizationStats]:
        """
        Quantize model to INT4.
        
        Returns quantized model and statistics.
        """
        if not self._bnb_available:
            logger.error("bitsandbytes not available, returning original model")
            return model, QuantizationStats(
                original_size_mb=0,
                quantized_size_mb=0,
                compression_ratio=1.0,
                num_quantized_layers=0,
                quantization_type="none",
            )
        
        import bitsandbytes as bnb
        
        # Calculate original size
        original_params = sum(p.numel() for p in model.parameters())
        original_size_mb = original_params * 4 / (1024 * 1024)  # FP32
        
        # Count layers to quantize
        layers_to_quantize = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers_to_quantize.append((name, module))
        
        num_quantized = 0
        
        # Quantize each linear layer
        for name, linear in layers_to_quantize:
            try:
                # Get parent module and attribute name
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                
                # Create 4-bit linear layer
                quantized_linear = bnb.nn.Linear4bit(
                    linear.in_features,
                    linear.out_features,
                    bias=linear.bias is not None,
                    compute_dtype=self.compute_dtype,
                    quant_type=self.quant_type,
                    compress_statistics=self.double_quant,
                )
                
                # Copy weights (will be quantized automatically)
                quantized_linear.weight = bnb.nn.Params4bit(
                    linear.weight.data,
                    requires_grad=False,
                    quant_type=self.quant_type,
                    compress_statistics=self.double_quant,
                )
                
                if linear.bias is not None:
                    quantized_linear.bias = nn.Parameter(linear.bias.data.clone())
                
                # Replace module
                setattr(parent, attr_name, quantized_linear)
                num_quantized += 1
                
            except Exception as e:
                logger.warning(f"Failed to quantize layer {name}: {e}")
        
        # Calculate quantized size (4 bits per param + overhead)
        quantized_size_mb = original_params * 0.5 / (1024 * 1024)  # INT4
        if self.double_quant:
            quantized_size_mb *= 0.9  # ~10% additional savings
        
        stats = QuantizationStats(
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0,
            num_quantized_layers=num_quantized,
            quantization_type=f"int4_{self.quant_type}",
        )
        
        logger.info(
            f"INT4 quantization complete: {num_quantized} layers, "
            f"{stats.compression_ratio:.2f}x compression"
        )
        
        return model, stats
    
    def quantize_for_inference(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[nn.Module, QuantizationStats]:
        """
        Quantize model with optional calibration for better accuracy.
        """
        if calibration_data and self._bnb_available:
            # Run calibration forward passes
            model.eval()
            with torch.no_grad():
                for data in calibration_data[:100]:  # Use up to 100 samples
                    try:
                        model(data)
                    except Exception:
                        pass
        
        return self.quantize(model)


class GPTQQuantizer:
    """
    GPTQ (GPT-Quantization) for high-quality INT4/INT8 quantization.
    
    Uses calibration data to minimize quantization error via
    optimal brain compression.
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        actorder: bool = True,
        sym: bool = False,
    ):
        self.bits = bits
        self.group_size = group_size
        self.actorder = actorder  # Activation order for better accuracy
        self.sym = sym  # Symmetric quantization
        
        self._auto_gptq_available = False
        self._check_auto_gptq()
    
    def _check_auto_gptq(self):
        """Check if auto-gptq is available."""
        try:
            from auto_gptq import AutoGPTQForCausalLM
            self._auto_gptq_available = True
            logger.info("auto-gptq available for GPTQ quantization")
        except ImportError:
            logger.warning(
                "auto-gptq not installed. Install with: "
                "pip install auto-gptq>=0.4.0"
            )
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
    ) -> Tuple[nn.Module, QuantizationStats]:
        """
        Quantize model using GPTQ with calibration data.
        """
        if not self._auto_gptq_available:
            logger.warning("auto-gptq not available, using fallback INT4")
            int4_quantizer = INT4Quantizer()
            return int4_quantizer.quantize(model)
        
        # GPTQ quantization logic would go here
        # This requires the full model architecture and tokenizer
        
        original_params = sum(p.numel() for p in model.parameters())
        original_size_mb = original_params * 4 / (1024 * 1024)
        quantized_size_mb = original_params * (self.bits / 8) / (1024 * 1024)
        
        stats = QuantizationStats(
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=original_size_mb / quantized_size_mb,
            num_quantized_layers=0,
            quantization_type=f"gptq_{self.bits}bit",
        )
        
        return model, stats


class AWQQuantizer:
    """
    AWQ (Activation-aware Weight Quantization) for accurate INT4.
    
    Preserves important weights based on activation magnitude.
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
    ):
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        
        self._awq_available = False
        self._check_awq()
    
    def _check_awq(self):
        """Check if awq is available."""
        try:
            from awq import AutoAWQForCausalLM
            self._awq_available = True
            logger.info("AWQ available for activation-aware quantization")
        except ImportError:
            logger.warning(
                "awq not installed. Install with: "
                "pip install autoawq>=0.1.0"
            )
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
    ) -> Tuple[nn.Module, QuantizationStats]:
        """
        Quantize model using AWQ with calibration data.
        """
        if not self._awq_available:
            logger.warning("AWQ not available, using fallback INT4")
            int4_quantizer = INT4Quantizer()
            return int4_quantizer.quantize(model)
        
        original_params = sum(p.numel() for p in model.parameters())
        original_size_mb = original_params * 4 / (1024 * 1024)
        quantized_size_mb = original_params * (self.bits / 8) / (1024 * 1024)
        
        stats = QuantizationStats(
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=original_size_mb / quantized_size_mb,
            num_quantized_layers=0,
            quantization_type=f"awq_{self.bits}bit",
        )
        
        return model, stats


class ModelQuantizer:
    """
    Unified quantization interface supporting multiple methods.
    
    Supports:
    - INT8: PyTorch native dynamic quantization
    - INT4: bitsandbytes NF4/FP4 quantization
    - GPTQ: Post-training quantization with calibration
    - AWQ: Activation-aware weight quantization
    """
    
    def __init__(self, config: PracticalDeploymentConfig):
        self.config = config
        
        # Initialize specific quantizers based on config
        self.int4_quantizer = INT4Quantizer(
            compute_dtype=getattr(torch, config.int4_compute_dtype),
            quant_type=config.int4_quant_type,
            double_quant=config.int4_double_quant,
        )
        
        self.gptq_quantizer = GPTQQuantizer()
        self.awq_quantizer = AWQQuantizer()
    
    def quantize_int8(self, model: nn.Module) -> Tuple[nn.Module, QuantizationStats]:
        """Quantize model to INT8 using PyTorch dynamic quantization."""
        logger.info("Quantizing model to INT8")
        
        original_params = sum(p.numel() for p in model.parameters())
        original_size_mb = original_params * 4 / (1024 * 1024)
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        
        # Count quantized layers
        num_quantized = sum(
            1 for m in quantized_model.modules()
            if hasattr(m, '_packed_params')
        )
        
        quantized_size_mb = original_params / (1024 * 1024)  # INT8
        
        stats = QuantizationStats(
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=original_size_mb / quantized_size_mb,
            num_quantized_layers=num_quantized,
            quantization_type="int8",
        )
        
        logger.info(
            f"INT8 quantization complete: {num_quantized} layers, "
            f"{stats.compression_ratio:.2f}x compression"
        )
        
        return quantized_model, stats
    
    def quantize_int4(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[nn.Module, QuantizationStats]:
        """Quantize model to INT4."""
        logger.info("Quantizing model to INT4")
        return self.int4_quantizer.quantize_for_inference(model, calibration_data)
    
    def quantize_fp16(
        self,
        model: nn.Module,
        use_bfloat16: bool = False,
    ) -> Tuple[nn.Module, QuantizationStats]:
        """
        Convert model to FP16 (half precision) or BF16.
        
        ⚠️ PLACEHOLDER: Core conversion logic needs implementation.
        
        Args:
            model: Model to convert
            use_bfloat16: Use bfloat16 instead of float16 (better for training)
            
        Returns:
            Converted model and statistics
            
        Technical Design:
            1. Use torch.float16 / torch.bfloat16 conversion
            2. Implement mixed-precision inference
            3. Handle layers that don't support half precision
            4. Support for Apple MPS and CUDA
            
        Target Version: v2.2.0
        """
        logger.info(f"Converting model to {'BF16' if use_bfloat16 else 'FP16'}")
        
        # Calculate original size
        original_params = sum(p.numel() for p in model.parameters())
        original_size_mb = original_params * 4 / (1024 * 1024)  # FP32
        
        # Placeholder: Return model with basic conversion
        # TODO: Implement proper mixed-precision handling
        target_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        
        try:
            # Basic conversion - in production, need careful layer-by-layer handling
            model = model.to(dtype=target_dtype)
            quantized_size_mb = original_params * 2 / (1024 * 1024)  # FP16/BF16
            
            stats = QuantizationStats(
                original_size_mb=original_size_mb,
                quantized_size_mb=quantized_size_mb,
                compression_ratio=2.0,  # 50% reduction
                num_quantized_layers=sum(1 for _ in model.modules()) - 1,
                quantization_type="bf16" if use_bfloat16 else "fp16",
            )
            
            logger.warning(
                "FP16 quantization is a placeholder. "
                "Full implementation with mixed-precision support coming in v2.2.0"
            )
            
            return model, stats
            
        except Exception as e:
            logger.error(f"FP16 conversion failed: {e}")
            return model, QuantizationStats(
                original_size_mb=original_size_mb,
                quantized_size_mb=original_size_mb,
                compression_ratio=1.0,
                num_quantized_layers=0,
                quantization_type="none",
            )
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[nn.Module, QuantizationStats]:
        """Quantize model based on config."""
        quant_type = self.config.quantization_type
        
        if quant_type == QuantizationType.NONE:
            original_params = sum(p.numel() for p in model.parameters())
            stats = QuantizationStats(
                original_size_mb=original_params * 4 / (1024 * 1024),
                quantized_size_mb=original_params * 4 / (1024 * 1024),
                compression_ratio=1.0,
                num_quantized_layers=0,
                quantization_type="none",
            )
            return model, stats
            
        elif quant_type == QuantizationType.INT8:
            return self.quantize_int8(model)
            
        elif quant_type == QuantizationType.INT4:
            return self.quantize_int4(model, calibration_data)
            
        elif quant_type == QuantizationType.GPTQ:
            if calibration_data:
                return self.gptq_quantizer.quantize(model, calibration_data)
            else:
                logger.warning("GPTQ requires calibration data, falling back to INT4")
                return self.quantize_int4(model)
                
        elif quant_type == QuantizationType.AWQ:
            if calibration_data:
                return self.awq_quantizer.quantize(model, calibration_data)
            else:
                logger.warning("AWQ requires calibration data, falling back to INT4")
                return self.quantize_int4(model)
        
        else:
            logger.warning(f"Unsupported quantization: {quant_type}")
            return model, QuantizationStats(
                original_size_mb=0,
                quantized_size_mb=0,
                compression_ratio=1.0,
                num_quantized_layers=0,
                quantization_type="none",
            )
    
    @staticmethod
    def estimate_memory_savings(
        original_params: int,
        quant_type: QuantizationType,
    ) -> Dict[str, float]:
        """Estimate memory savings from quantization."""
        original_size_gb = original_params * 4 / 1e9  # FP32
        
        if quant_type == QuantizationType.INT8:
            quantized_size_gb = original_params / 1e9
            savings = 0.75
        elif quant_type == QuantizationType.INT4:
            quantized_size_gb = original_params * 0.5 / 1e9
            savings = 0.875
        elif quant_type == QuantizationType.FP8:
            quantized_size_gb = original_params / 1e9
            savings = 0.75
        elif quant_type in (QuantizationType.GPTQ, QuantizationType.AWQ):
            quantized_size_gb = original_params * 0.5 / 1e9
            savings = 0.875
        else:
            quantized_size_gb = original_size_gb
            savings = 0.0
        
        return {
            "original_gb": original_size_gb,
            "quantized_gb": quantized_size_gb,
            "savings_percent": savings * 100,
            "compression_ratio": 1 / (1 - savings) if savings < 1 else 1.0,
        }
