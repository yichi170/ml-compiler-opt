
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Dummy quantile map for demonstration.
# In a real implementation, this would come from feature_ops.build_quantile_map
# and contain actual quantile values for each feature.
DUMMY_QUANTILE_MAP = {
    'pos': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'excess': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'current_max': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'critical_max': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'su_latency': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'su_height': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'su_depth': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'su_succs_left': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'su_preds_left': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'su_succs': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'su_preds': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'sgpr_critical_limit': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'vgpr_critical_limit': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'sgpr_excess_limit': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'vgpr_excess_limit': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
}

class NormalizationLayer(nn.Module):
    def __init__(self, quantile, with_sqrt=True, with_z_score_normalization=True, eps=1e-8, preprocessing_fn=None):
        super(NormalizationLayer, self).__init__()
        self.quantile = torch.tensor(quantile, dtype=torch.float32)
        self.with_sqrt = with_sqrt
        self.with_z_score_normalization = with_z_score_normalization
        self.eps = eps
        self.preprocessing_fn = preprocessing_fn

        self.first_non_zero = 0.0
        for x in self.quantile:
            if x > 0:
                self.first_non_zero = x
                break

    def forward(self, x):
        if self.preprocessing_fn:
            x = self.preprocessing_fn(x)

        max_quantile = self.quantile.max()
        if max_quantile > 0:
            x = x / max_quantile
        
        if self.with_sqrt:
            x = torch.sqrt(x + self.eps)

        if self.with_z_score_normalization:
            pass 
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, num_candidates=256):
        super(FeatureExtractor, self).__init__()
        self.num_candidates = num_candidates

        self.mask_layer = nn.Identity()
        self.is_top_layer = nn.Identity()
        self.is_bot_layer = nn.Identity()

        self.pos_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['pos'])
        self.excess_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['excess'])
        self.current_max_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['current_max'])
        self.critical_max_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['critical_max'])
        self.su_succs_left_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['su_succs_left'])
        self.su_preds_left_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['su_preds_left'])
        self.su_succs_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['su_succs'])
        self.su_preds_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['su_preds'])

        self.su_latency_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['su_latency'], preprocessing_fn=lambda x: torch.log(x + DUMMY_QUANTILE_MAP['su_latency'][0] if DUMMY_QUANTILE_MAP['su_latency'][0] > 0 else x + 1e-8))
        self.su_height_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['su_height'], preprocessing_fn=lambda x: torch.log(x + DUMMY_QUANTILE_MAP['su_height'][0] if DUMMY_QUANTILE_MAP['su_height'][0] > 0 else x + 1e-8))
        self.su_depth_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['su_depth'], preprocessing_fn=lambda x: torch.log(x + DUMMY_QUANTILE_MAP['su_depth'][0] if DUMMY_QUANTILE_MAP['su_depth'][0] > 0 else x + 1e-8))

        self.sgpr_critical_limit_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['sgpr_critical_limit'])
        self.vgpr_critical_limit_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['vgpr_critical_limit'])
        self.sgpr_excess_limit_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['sgpr_excess_limit'])
        self.vgpr_excess_limit_layer = NormalizationLayer(DUMMY_QUANTILE_MAP['vgpr_excess_limit'])

        self.output_dim = 18 * self.num_candidates

    def forward(self, observations):
        processed_features = []
        processed_features.append(self.mask_layer(observations['mask'].float()))
        processed_features.append(self.is_top_layer(observations['is_top'].float()))
        processed_features.append(self.is_bot_layer(observations['is_bot'].float()))
        
        processed_features.append(self.pos_layer(observations['pos'].float()))
        processed_features.append(self.excess_layer(observations['excess'].float()))
        processed_features.append(self.current_max_layer(observations['current_max'].float()))
        processed_features.append(self.critical_max_layer(observations['critical_max'].float()))
        processed_features.append(self.su_succs_left_layer(observations['su_succs_left'].float()))
        processed_features.append(self.su_preds_left_layer(observations['su_preds_left'].float()))
        processed_features.append(self.su_succs_layer(observations['su_succs'].float()))
        processed_features.append(self.su_preds_layer(observations['su_preds'].float()))

        processed_features.append(self.su_latency_layer(observations['su_latency'].float()))
        processed_features.append(self.su_height_layer(observations['su_height'].float()))
        processed_features.append(self.su_depth_layer(observations['su_depth'].float()))

        sgpr_critical_limit = self.sgpr_critical_limit_layer(observations['sgpr_critical_limit'].float()).unsqueeze(-1).expand(-1, self.num_candidates)
        vgpr_critical_limit = self.vgpr_critical_limit_layer(observations['vgpr_critical_limit'].float()).unsqueeze(-1).expand(-1, self.num_candidates)
        sgpr_excess_limit = self.sgpr_excess_limit_layer(observations['sgpr_excess_limit'].float()).unsqueeze(-1).expand(-1, self.num_candidates)
        vgpr_excess_limit = self.vgpr_excess_limit_layer(observations['vgpr_excess_limit'].float()).unsqueeze(-1).expand(-1, self.num_candidates)

        processed_features.append(sgpr_critical_limit)
        processed_features.append(vgpr_critical_limit)
        processed_features.append(sgpr_excess_limit)
        processed_features.append(vgpr_excess_limit)

        combined_features = torch.stack(processed_features, dim=2)
        flattened_features = combined_features.view(combined_features.size(0), -1)
        return flattened_features
