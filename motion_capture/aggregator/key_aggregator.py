import numpy as np
from xrmocap.data_structure.body_model.smplx_data import SMPLXData

from .base_aggregator import BaseAggregator


class KeyAggregator(BaseAggregator):

    def __init__(self, mapping_dict, logger=None) -> None:
        BaseAggregator.__init__(self, logger=logger)
        self.mapping_dict = mapping_dict
        self.last_global_orient = None

    def forward(self, smplx_nested_dict, **kwargs):
        ret_dict = {}
        # for key, selected_key_list in self.mapping_dict.items():
        #     for selected_key in selected_key_list:
        #         ret_dict[selected_key] = smplx_nested_dict[key][selected_key]
        if self.last_global_orient is None:
            global_orient = np.zeros(shape=(1, 1, 3))
        else:
            offset = [(np.random.rand() - 0.5), (np.random.rand() - 0.5),
                      (np.random.rand() - 0.5)]
            offset = np.array(offset).reshape(1, 1, 3) * 0.25
            global_orient = self.last_global_orient + offset
        fullpose = np.concatenate((
            global_orient,
            np.zeros(shape=(1, 21, 3)),
            np.zeros(shape=(1, 3, 3)),
            np.zeros(shape=(1, 15, 3)),
            np.zeros(shape=(1, 15, 3)),
        ),
                                  axis=1)
        smplx_data = SMPLXData(
            fullpose=fullpose,
            transl=np.array([[0.17405687, 0.76408563, 1.43093603]]),
            betas=np.zeros(shape=(1, 10)))
        ret_dict['smplx_data'] = smplx_data
        self.last_global_orient = global_orient
        return ret_dict
