from torchvision.datasets.cifar import CIFAR10
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FLD import FLD
from dataclasses import dataclass
from typing import Any
from fld.metrics.FID import FID
from fld.metrics.PrecisionRecall import PrecisionRecall

@dataclass
class Metric:
    path_to_cifar_images: str
    path_to_gen_images: str
    img_extension: str
    train_feat: Any
    test_feat: Any
    gen_feat: Any
    output_file: str = "./metrics"
    feature_extractor: InceptionFeatureExtractor = InceptionFeatureExtractor()

    def compute_FID(self):
        train_feat = self.feature_extractor.get_features(
            CIFAR10(train=True, root=self.path_to_cifar_images, download=False)
        )
        gen_feat = self.feature_extractor.get_dir_features(
            self.path_to_gen_images, extension=self.img_extension
        )
        fid_val = FID().compute_metric(train_feat, None, gen_feat)
        precision = PrecisionRecall(mode="Precision").compute_metric(train_feat, None, gen_feat) # Default precision
        recall = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(train_feat, None, gen_feat) # Recall with k=5
        with open(self.output_file, "a") as f:
            f.write(str([fid_val, precision, recall]))



