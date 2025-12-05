# Foundation Model Deployment Guide

## Progression Path: Plan A → Plan B

```
Phase 1: Plan A (Production)     →     Phase 2: Plan B (Research)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ┌─────────────────────────┐         ┌─────────────────────────┐
 │    Frozen Base Model    │         │   Full MoE Transformer  │
 │    + LoRA Adapters      │   ───►  │   500B-1T Parameters    │
 │    + RAG System         │         │   + Autonomous Learning │
 └─────────────────────────┘         └─────────────────────────┘

 Cost: $1-5M/year                    Cost: $10-100M/year
 Hardware: 1-8 GPUs                  Hardware: 64-1024 GPUs
 Setup: Hours                        Setup: Weeks
```

---

## Phase 1: Plan A - Production Deployment

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
cd ai_core/foundation_model
pip install -r requirements.txt

# For INT4 quantization (optional)
pip install bitsandbytes

# For production inference
pip install vllm
```

### Step 2: Load Base Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose your base model (examples)
BASE_MODELS = {
    "small": "mistralai/Mistral-7B-v0.1",      # 7B params
    "medium": "meta-llama/Llama-2-13b-hf",    # 13B params
    "large": "meta-llama/Llama-2-70b-hf",     # 70B params
}

# Load model
model_name = BASE_MODELS["small"]  # Start small
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

### Step 3: Initialize Plan A System

```python
from ai_core.foundation_model import (
    PracticalDeploymentSystem,
    PracticalDeploymentConfig,
    RetrainingFrequency,
    QuantizationType,
)

# Configure for production
config = PracticalDeploymentConfig(
    # Base model settings
    freeze_base_model=True,

    # LoRA configuration
    lora_r=8,                    # Low rank (8-64)
    lora_alpha=32,               # Scaling factor
    lora_dropout=0.05,           # Regularization
    lora_target_modules=[        # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # RAG configuration
    enable_rag=True,
    rag_top_k=5,                 # Retrieve top 5 documents
    rag_similarity_threshold=0.7,
    rag_index_path="./data/rag_index",

    # Retraining schedule
    retraining_frequency=RetrainingFrequency.WEEKLY,
    retraining_data_path="./data/retraining",
    min_samples_for_retraining=1000,

    # Quantization (for efficiency)
    quantization_type=QuantizationType.INT8,

    # Cost control
    max_daily_tokens=100_000_000,      # 100M tokens/day
    max_monthly_cost_usd=50_000,       # $50K/month

    # Fault tolerance
    checkpoint_interval_minutes=30,
    max_retries=3,
    health_check_interval_seconds=60,
)

# Create and start system
system = PracticalDeploymentSystem(model, config)
```

### Step 4: Start Production Service

```python
import asyncio

async def main():
    # Start the system
    await system.start()
    print("✅ Plan A system started")

    # Check status
    status = system.get_status()
    print(f"Active adapter: {status['active_adapter']}")
    print(f"RAG enabled: {config.enable_rag}")
    print(f"Next retraining: {status['retraining']['next']}")

asyncio.run(main())
```

### Step 5: Add Knowledge to RAG

```python
# Add code review best practices
system.add_knowledge(
    content="""
    Security Best Practices for Python:
    1. Never use eval() with user input
    2. Use parameterized queries for SQL
    3. Validate all input data
    4. Use secrets module for tokens
    5. Enable HTTPS for all connections
    """,
    metadata={"source": "OWASP", "category": "security", "language": "python"}
)

# Add from documentation files
import os
for doc_file in os.listdir("./docs/knowledge"):
    with open(f"./docs/knowledge/{doc_file}") as f:
        system.add_knowledge(
            content=f.read(),
            metadata={"source": doc_file}
        )

# Save RAG index
system.rag_system.index.save()
print(f"✅ Added {len(system.rag_system.index.documents)} documents to RAG")
```

### Step 6: Process Requests

````python
async def process_code_review(code: str, language: str = "python"):
    """Process a code review request with RAG augmentation."""

    prompt = f"""Review the following {language} code for:
- Security vulnerabilities
- Performance issues
- Best practice violations
- Code quality problems

```{language}
{code}
````

Provide detailed analysis with severity levels."""

    result = await system.process(
        input_text=prompt,
        use_rag=True,  # Enable RAG augmentation
    )

    return result

# Example usage

code = '''
def get_user(user_id):
query = f"SELECT \* FROM users WHERE id = {user_id}"
return db.execute(query)
'''

result = await process_code_review(code)
print(result["output"])
print(f"RAG context used: {len(result['rag_context'])} documents")

````

### Step 7: Create Domain-Specific Adapters

```python
# Create adapters for different domains
adapter_manager = system.adapter_manager

# Security-focused adapter
adapter_manager.create_adapter("security")

# Performance-focused adapter
adapter_manager.create_adapter("performance")

# Code quality adapter
adapter_manager.create_adapter("quality")

# Switch adapters based on task
adapter_manager.activate_adapter("security")
result = await system.process("Check for SQL injection...")

adapter_manager.activate_adapter("performance")
result = await system.process("Optimize this loop...")
````

### Step 8: Collect Training Data

```python
# The system automatically collects data
# You can also add samples manually with feedback

system.data_collector.add_sample(
    input_text="Review this code for SQL injection...",
    output_text="Found SQL injection vulnerability at line 5...",
    metadata={"category": "security", "quality": "high"},
    uncertainty=0.1,  # Low uncertainty = high confidence
)

# Check if ready for retraining
if system.data_collector.is_ready_for_retraining():
    print(f"Ready! {system.data_collector.sample_count} samples collected")
```

### Step 9: Trigger Retraining

```python
# Automatic (scheduled)
# The RetrainingScheduler runs automatically based on config

# Manual trigger
await system.trigger_retraining(adapter_name="security")

# Check retraining history
for entry in system.retraining_scheduler.training_history:
    print(f"Trained {entry['adapter']} at {entry['ended_at']}")
    print(f"  Samples: {entry['samples']}, Loss: {entry['avg_loss']:.4f}")
```

### Step 10: Monitor & Scale

```python
# Get system status
status = system.get_status()
print(f"""
System Status:
  Running: {status['is_running']}
  Health: {status['health']['status']}

Cost Usage:
  Daily tokens: {status['cost']['daily_tokens']:,} / {status['cost']['daily_limit']:,}
  Monthly cost: ${status['cost']['monthly_cost']:.2f} / ${status['cost']['monthly_limit']:.2f}

Retraining:
  Last: {status['retraining']['last']}
  Next: {status['retraining']['next']}
  Samples collected: {status['data_collected']}
""")
```

---

## Phase 2: Plan B - Research Evolution

### When to Evolve to Plan B

Consider evolving when:

- ✅ Plan A running stable for 6+ months
- ✅ Need capabilities beyond RAG (reasoning, planning)
- ✅ Budget increased to $10M+/year
- ✅ Have 64+ GPU cluster available
- ✅ Need domain-specific pre-training
- ✅ Want self-improving capabilities

### Step 1: Infrastructure Upgrade

```yaml
# kubernetes/plan-b-cluster.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: foundation-model-research
---
# GPU Node Pool
apiVersion: container.google.com/v1
kind: NodePool
metadata:
  name: gpu-pool
spec:
  nodeCount: 16 # 16 nodes × 8 GPUs = 128 GPUs
  machineType: a2-ultragpu-8g # 8x A100 80GB per node
  accelerators:
    - type: nvidia-tesla-a100
      count: 8
```

### Step 2: Initialize MoE Model

```python
from ai_core.foundation_model import (
    MoETransformer,
    MoEConfig,
    get_moe_config_500b,
)

# Create 500B parameter MoE model
config = get_moe_config_500b()
# Or customize:
config = MoEConfig(
    vocab_size=128000,
    hidden_size=8192,
    num_hidden_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,           # GQA
    num_experts=128,                 # Total experts
    num_experts_per_token=2,         # Active per token
    max_position_embeddings=131072,  # 128K context
    use_flash_attention=True,
    use_sparse_attention=True,
)

model = MoETransformer(config)
print(f"Total parameters: {config.total_params:,}")
print(f"Active parameters: {config.active_params:,}")
```

### Step 3: Pre-train on Domain Data

```python
from ai_core.foundation_model import (
    PretrainingEngine,
    PretrainingConfig,
    DataPipeline,
)

# Configure pre-training
pretrain_config = PretrainingConfig(
    learning_rate=3e-4,
    warmup_steps=2000,
    max_steps=500000,
    global_batch_size=4096,
    precision="bf16",
    gradient_accumulation_steps=32,

    # 4D Parallelism
    tensor_parallel_size=8,
    pipeline_parallel_size=4,
    data_parallel_size=4,

    # Checkpointing
    checkpoint_dir="./checkpoints/planb",
    save_interval=1000,
)

# Create training engine
engine = PretrainingEngine(
    config=pretrain_config,
    model=model,
    train_dataloader=train_loader,
)

# Start training
state = engine.train(resume=True)
```

### Step 4: Post-Training Alignment

```python
from ai_core.foundation_model import (
    ValueAligner,
    PosttrainingConfig,
    AlignmentMethod,
)

# Configure alignment
align_config = PosttrainingConfig(
    method=AlignmentMethod.RLHF,  # or DPO, CONSTITUTIONAL

    # SFT settings
    sft_epochs=3,
    sft_learning_rate=2e-5,

    # RLHF settings
    ppo_epochs=4,
    ppo_clip_ratio=0.2,
    kl_target=0.02,
)

# Run alignment
aligner = ValueAligner(model, align_config, tokenizer)
results = aligner.align(
    sft_data=sft_conversations,
    preference_data=preference_pairs,
    prompts=rl_prompts,
)
```

### Step 5: Enable Autonomous Learning

```python
from ai_core.foundation_model import (
    AutonomousLearningAgent,
    AutonomousConfig,
    SafetyLevel,
)

# Configure autonomous learning
auto_config = AutonomousConfig(
    # Online learning
    primary_mode="online",
    online_learning_rate=1e-6,
    online_buffer_size=10000,

    # Memory system
    episodic_memory_size=100000,
    working_memory_size=1000,
    enable_memory_consolidation=True,
    consolidation_interval_hours=24,

    # Self-evaluation
    eval_interval_steps=1000,
    benchmark_suite=["code_review", "security", "performance"],
    knowledge_gap_threshold=0.1,

    # Safety (CRITICAL)
    safety_level=SafetyLevel.HIGH,
    human_oversight_required=True,
    max_autonomous_steps=10000,
)

# Create and start agent
agent = AutonomousLearningAgent(model, auto_config)
await agent.start()

# Monitor
status = agent.get_status()
print(f"Autonomous steps: {status['autonomous_steps']}")
print(f"Knowledge gaps: {status['knowledge_gaps']}")
print(f"Safety status: {status['safety']}")
```

### Step 6: Integrate with Three-Version System

```python
from ai_core.foundation_model import (
    create_enhanced_coordinator,
    ModelTier,
)

# Create enhanced coordinator with Plan B capabilities
coordinator = create_enhanced_coordinator(
    tier=ModelTier.ENTERPRISE,  # Full capabilities
)

# Route requests through three-version system
# V1 (Experimental) - Test new models
# V2 (Production) - Serve users
# V3 (Quarantine) - Failed experiments

result = await coordinator.route_request(
    user_role="user",
    request_type="code_review",
    request_data={"code": code, "language": "python"},
)
```

---

## Cost Comparison

| Phase                 | Hardware   | Monthly Cost | Annual Cost |
| --------------------- | ---------- | ------------ | ----------- |
| **Plan A Small**      | 1x A100    | ~$4K         | ~$50K       |
| **Plan A Medium**     | 8x A100    | ~$30K        | ~$360K      |
| **Plan A Large**      | 16x A100   | ~$60K        | ~$720K      |
| **Plan B Entry**      | 64x H100   | ~$500K       | ~$6M        |
| **Plan B Full**       | 256x H100  | ~$2M         | ~$24M       |
| **Plan B Enterprise** | 1024x H100 | ~$8M         | ~$100M      |

---

## Migration Checklist

### Before Migration (Plan A → Plan B)

- [ ] Plan A stable for 6+ months
- [ ] Collected 1M+ high-quality samples
- [ ] Budget approved ($10M+/year)
- [ ] GPU cluster provisioned (64+ GPUs)
- [ ] Team trained on distributed systems
- [ ] Safety protocols documented
- [ ] Rollback plan prepared

### During Migration

- [ ] Export Plan A adapters and RAG index
- [ ] Initialize Plan B infrastructure
- [ ] Pre-train on collected data
- [ ] Run alignment (RLHF/DPO)
- [ ] Enable autonomous learning (supervised)
- [ ] A/B test against Plan A
- [ ] Gradual traffic migration

### After Migration

- [ ] Monitor safety metrics daily
- [ ] Weekly human review of autonomous decisions
- [ ] Monthly capability benchmarks
- [ ] Quarterly cost review
- [ ] Keep Plan A as fallback

---

## Quick Reference

### Plan A Commands

```python
# Start
system = PracticalDeploymentSystem(model, config)
await system.start()

# Process
result = await system.process(prompt, use_rag=True)

# Add knowledge
system.add_knowledge(content, metadata)

# Switch adapter
system.adapter_manager.activate_adapter("security")

# Trigger retraining
await system.trigger_retraining()

# Get status
status = system.get_status()

# Stop
await system.stop()
```

### Plan B Commands

```python
# Start
agent = AutonomousLearningAgent(model, config)
await agent.start()

# Process
result = await agent.process_input(prompt, embedding)

# Add learning sample
agent.add_learning_sample(sample, priority=1.0)

# Get status
status = agent.get_status()

# Emergency stop
agent.safety.trigger_emergency_stop("reason")

# Stop
await agent.stop()
```

---

## Recommended Timeline

```
Month 1-3:   Plan A Setup & Testing
Month 4-6:   Plan A Production, Data Collection
Month 7-12:  Plan A Optimization, Evaluate Plan B Need
Month 13-15: Plan B Infrastructure Setup (if needed)
Month 16-18: Plan B Training & Alignment
Month 19-24: Plan B Gradual Rollout
```

**Start with Plan A. It's production-ready today.** 🚀
