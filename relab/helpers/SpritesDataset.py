import collections
import numpy as np
import os
from os.path import join, exists
import invoke


SpritesDataset = collections.namedtuple("SpritesDataset", field_names=["images", "s_sizes", "s_dim", "s_bases"])


class DataSet:
    """!
    Singleton used to access the dSprites dataset.
    """

    ## @var instance
    # The unique instance of the d-sprites dataset (singleton design pattern).
    instance = None

    @staticmethod
    def get() -> SpritesDataset:
        """!
        Getter.
        @return an object containing the dSprite dataset
        """

        # Download the d-sprites dataset, if the function is called for the first time.
        dataset_dir = join(os.environ["DATASET_DIRECTORY"], "d_sprites")
        images_archive = join(dataset_dir, "d_sprites.npz")
        if not exists(dataset_dir):
            repository_dir = join(dataset_dir, "dsprites-dataset")
            old_images_archive = join(repository_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
            invoke.run(
                f"mkdir -p {dataset_dir} && cd {dataset_dir}"
                f"&& git clone https://github.com/google-deepmind/dsprites-dataset.git"
                f"&& mv {old_images_archive} {images_archive} && rm -rf {repository_dir}"
            )

        # Load the d-sprites dataset.
        if DataSet.instance is None:
            dataset = np.load(images_archive, allow_pickle=True, encoding="latin1")
            images = dataset["imgs"].reshape(-1, 64, 64, 1)
            metadata = dataset["metadata"][()]
            s_sizes = metadata["latents_sizes"]  # [1 3 6 40 32 32]
            s_dim = s_sizes.size
            s_bases = np.concatenate((metadata["latents_sizes"][::-1].cumprod()[::-1][1:], np.array([1, ])))
            s_bases = np.squeeze(s_bases)  # self.s_bases = [737280 245760  40960 1024 32]
            DataSet.instance = SpritesDataset(images, s_sizes, s_dim, s_bases)
        return DataSet.instance