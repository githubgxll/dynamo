# SPDX-License-Identifier: Apache-2.0
"""Bounded internal diagnostics, deliberately separate from public task errors."""
from collections.abc import Mapping

REASONS = frozenset({'resource_exhausted', 'engine_unavailable', 'input_processing_failed', 'unknown'})
STAGES = frozenset({'input_processing', 'base_generation', 'super_resolution', 'result_validation', 'worker_execution', 'unknown'})

def normalize_internal_error(value, *, stage='unknown'):
    value = value if isinstance(value, Mapping) and value.get('schema_version') == 1 else {}
    return {
        'schema_version': 1,
        'reason': value.get('reason') if isinstance(value.get('reason'), str) and value.get('reason') in REASONS else 'unknown',
        'stage': value.get('stage') if isinstance(value.get('stage'), str) and value.get('stage') in STAGES else (stage if stage in STAGES else 'unknown'),
    }

class VideoWorkerFailure(RuntimeError):
    def __init__(self, message, internal_error=None, *, stage='unknown'):
        super().__init__(message)
        self.internal_error = normalize_internal_error(internal_error, stage=stage)

def classify_worker_error(exc, *, stage='worker_execution'):
    if isinstance(exc, VideoWorkerFailure):
        return dict(exc.internal_error)
    reason = 'unknown'
    # Remote engines may serialize the concrete exception into a RuntimeError.
    # Match only explicit CUDA OOM signatures, never arbitrary RuntimeErrors.
    message = str(exc)
    types = {c.__name__ for c in type(exc).__mro__}
    if isinstance(exc, MemoryError) or 'OutOfMemoryError' in types or message.startswith('CUDA out of memory') or 'torch.OutOfMemoryError: CUDA out of memory' in message or 'torch.cuda.OutOfMemoryError: CUDA out of memory' in message:
        reason = 'resource_exhausted'
    elif 'EngineDeadError' in types or message in {'Executor shut down', 'Stage-0 has no live replica'}:
        reason = 'engine_unavailable'
    elif stage == 'input_processing' and isinstance(exc, (ValueError, TypeError)):
        reason = 'input_processing_failed'
    return normalize_internal_error({'schema_version': 1, 'reason': reason, 'stage': stage})

def failed_chunk_error(chunk, *, stage='worker_execution'):
    error = chunk.get('error')
    message = error.get('message') if isinstance(error, Mapping) else error
    # Missing metadata from an old Worker remains unknown, not inferred from
    # arbitrary protocol text received over the wire.
    return VideoWorkerFailure(str(message or 'Worker generation failed'), chunk.get('_internal_error'), stage=stage)
