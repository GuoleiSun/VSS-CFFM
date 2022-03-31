from .builder import DATASETS
from .custom import CustomDataset, CustomDataset_video2_vps


@DATASETS.register_module()
class IVPSDataset(CustomDataset):
    """IVPS dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'background', 'forward')

    PALETTE = [[0, 0, 0], [255,255,255]]

    def __init__(self, **kwargs):
        super(IVPSDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

@DATASETS.register_module()
class VPSDataset(CustomDataset_video2_vps):
    """IVPS dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'background', 'forward')

    PALETTE = [[0, 0, 0], [255,255,255]]

    def __init__(self, **kwargs):
        super(VPSDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
