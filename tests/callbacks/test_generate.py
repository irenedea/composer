from composer.callbacks import Generate
from composer.trainer import Trainer
from tests.common.models import configure_tiny_gpt2_hf_model
from tests.common.datasets import dummy_gpt_lm_dataloader
from tests.common.markers import device, world_size
from unittest.mock import Mock
import pytest
import torch
from packaging import version
import math


@device('cpu', 'gpu')
@pytest.mark.parametrize('use_fsdp', [True, False])
@world_size(1, 2)
def test_generate_callback(device, world_size, use_fsdp):
    if use_fsdp and version.parse(torch.__version__) < version.parse('1.13.0'):
        pytest.skip('FSDP requires torch >= 1.13.0')
    # transformers = pytest.importorskip('transformers')
    if device == 'cpu' and use_fsdp:
        pytest.skip('FSDP is not supported on CPU.')
    if world_size == 1 and use_fsdp:
        pytest.xfail((
            'Generation with world size 1 and FSDP fails with '
            '`RuntimeError: The tensor has a non-zero number of elements, '
            'but its data is not allocated yet. Caffe2 uses a lazy allocation, '
            'so you will need to call mutable_data() or raw_mutable_data() to actually allocate memory.` '
            'This issue is resolved with world size > 1 by a dummy call to forward (see HuggingFaceModel.dummy_forward_called), '
            'but for some reason fails with world size 1.'))

    # hf_model = hf_model()
    # if device == 'cpu' and world_size > 1 and isinstance(hf_model,
    #                                                      transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
    #     pytest.xfail(
    #         'GPT2 is not currently supported with DDP. See https://github.com/huggingface/transformers/issues/22482 for more details.'
    #     )

    model = configure_tiny_gpt2_hf_model()
    model.generate = Mock(wraps=model.generate)
    prompts = ['a','b','c']

    cb_batch_size = 2
    generate_cb = Generate(prompts, interval='1ba', batch_size=cb_batch_size, max_length=20)
    generate_cb.generate = Mock(wraps=generate_cb.generate)

    n_batches = 3
    trainer = Trainer(model=model, train_dataloader=dummy_gpt_lm_dataloader(), device=device, max_duration=f'{n_batches}ba',
                      callbacks=generate_cb)

    trainer.fit()

    print('call count:', model.generate.call_count, generate_cb.generate.call_count)

    assert generate_cb.generate.call_count == n_batches
    assert model.generate.call_count == math.ceil(len(prompts) / cb_batch_size) * n_batches * world_size
    



    
