from composer.callbacks import Generate
from composer.trainer import Trainer
from composer.utils import dist
from tests.common.models import configure_tiny_gpt2_hf_model
from tests.common.datasets import dummy_gpt_lm_dataloader
from tests.common.markers import device

@device('cpu', 'gpu')
def test_generate_callback(device):
    dist.initialize_dist(device=device)

    model = configure_tiny_gpt2_hf_model()

    trainer = Trainer(model=model, train_dataloader=dummy_gpt_lm_dataloader(), device=device, max_duration='3ba',
                      callbacks=Generate(['t1'], '1ba', batch_size=1, max_length=20))
    # TODO: assert that model.generate has been called the correct number of times on the correct batch

    trainer.fit()
