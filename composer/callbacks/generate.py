# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Periodically log generations to wandb from a set of prompts."""
from typing import Any, List, Union

from composer.core import Callback, Event, State, Time, get_precision_context
from composer.loggers import Logger
from composer.models import HuggingFaceModel
from composer.utils import dist
from composer.callbacks.checkpoint_saver import checkpoint_periodically

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch

from composer.utils.import_helpers import MissingConditionalImportError



class Generate(Callback):

    def __init__(self, prompts: List[str], interval: Union[str, int, Time], batch_size: int,
                 **kwargs: Any):
        """Periodically log generations.

        Args:
            prompts (List[str]): The list of prompts you would like to produce generations for
            interval (Union[str, int, :class:`.Time`]): The interval describing how often checkpoints should be
                saved. If an integer, it will be assumed to be in :attr:`.TimeUnit.EPOCH`.
                Otherwise, the unit must be either :attr:`.TimeUnit.EPOCH`, :attr:`.TimeUnit.BATCH`,
                :attr:`.TimeUnit.TOKEN`, or :attr:`.TimeUnit.SAMPLE`.
            kwargs: All kwargs well be passed along to the call to generate. This is for things like `do_sample`, `top_p`, etc
        """
        self.prompts = prompts
        self.generate_kwargs = kwargs
        self.batch_size = batch_size
        self.save_interval = checkpoint_periodically(interval, save_end_of_training=False)

    def init(self, state: State, logger: Logger):
        assert isinstance(state.model, HuggingFaceModel), f'Expected HuggingFaceModel got {state.model.__class__}'

    def run_event(self, event: Event, state: State, logger: Logger) -> None:
        if state.get_elapsed_duration() is not None and self.save_interval(state, event): 
            self.generate(state, logger)
        else:
            super().run_event(event, state, logger)

    def generate(self, state: State, logger: Logger):
        model = state.model
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(model, HuggingFaceModel) # TODO: Extend to support any models that have a generate method.

        tokenizer = model.tokenizer

        try: 
            from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        except ImportError as e: 
            raise MissingConditionalImportError(extra_deps_group='nlp', 
                                                conda_package='transformers', 
                                                conda_channel='conda-forge') from e 
        
        assert isinstance(tokenizer, Union[PreTrainedTokenizer, PreTrainedTokenizerFast])

        # Set to evaluation mode and stash the original mode.
        original_mode = model.training
        model.eval()
        device = state.device

        # Stash the original value of padding_side because generation requires left padding
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_input = tokenizer(self.prompts,
                                    return_tensors='pt',
                                    padding=True)

        for k, v in tokenized_input.items():
            tokenized_input[k] = device.tensor_to_device(v)

        all_input_ids = tokenized_input['input_ids']
        all_attn_masks = tokenized_input['attention_mask']

        batches = DataLoader(TensorDataset(all_input_ids, all_attn_masks), self.batch_size)

        output_token_ids = []
        for input_ids, attn_mask in batches:
            # Move batch to device.
            input_ids = device.tensor_to_device(input_ids)
            attn_mask = device.tensor_to_device(attn_mask)
            # dummy forward call needed for FSDP to work consistently
            dummy_input = torch.tensor([[0]], dtype=torch.long)
            dummy_input = device.tensor_to_device(dummy_input)
            with get_precision_context(state.precision):
                with torch.no_grad():
                    assert isinstance(model.model, torch.nn.Module)
                    _ = model.model(input_ids=dummy_input)
                output_token_ids.extend(model.generate(  # type: ignore
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    synced_gpus=dist.get_world_size() > 1,
                    **self.generate_kwargs,
                ))

        # output_token_ids = []
        #     # Move batch to device.
        # input_ids = device.tensor_to_device(all_input_ids)
        # attn_mask = device.tensor_to_device(all_attn_masks)
        # # dummy forward call needed for FSDP to work consistently
        # dummy_input = torch.tensor([[0]], dtype=torch.long)
        # dummy_input = device.tensor_to_device(dummy_input)
        # with get_precision_context(state.precision):
        #     with torch.no_grad():
        #         assert isinstance(model.model, torch.nn.Module)
        #         _ = model.model(input_ids=dummy_input)
        #     output_token_ids.extend(model.generate(  # type: ignore
        #         input_ids=input_ids,
        #         attention_mask=attn_mask,
        #         synced_gpus=dist.get_world_size() > 1,
        #         **self.generate_kwargs,
        #     ))
        if dist.get_global_rank() == 0:
            # Process prompts and outputs into a table.
            rows = []
            input_tokens_len = all_input_ids.shape[1]
            for i, prompt in enumerate(self.prompts):
                output_tokens = output_token_ids[i][input_tokens_len:]
                output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
                rows.append([prompt, output_text])

            # TODO: LOG TABLE

        tokenizer.padding_side = original_padding_side
        model.train(mode=original_mode)
