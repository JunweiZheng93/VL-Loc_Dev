from torch.utils.data import Dataset

class ContrastiveKittiDataset(Dataset):
    def __init__(self, anchor_dataset, positive_datasets, cfg):
        """
        A Dataset that pairs an anchor with its positive variations.

        Args:
            anchor_dataset (Dataset): The initialized dataset for clean anchor images.
            positive_datasets (dict[str, Dataset]): A dictionary of initialized datasets
                                                     for the noisy positive images.
        """
        super().__init__()
        self.anchor_dataset = anchor_dataset
        self.positive_datasets = positive_datasets
        self.cfg = cfg
        self.noise_types = list(self.positive_datasets.keys())

        # Sanity check: the number of noisy samples in each noisy dataset
        # should be the same to the number of samples in the anchor dataset.
        num_anchors = len(self.anchor_dataset)
        for noise, dataset in self.positive_datasets.items():
            assert len(dataset) == num_anchors, \
                f"Mismatch for '{noise}'. Anchor: {num_anchors}, Noisy: {len(dataset)}"

    def __len__(self):
        return len(self.anchor_dataset)

    def __getitem__(self, idx):
        """
        Fetches one anchor sample and all its corresponding positive images.
        """
        # 1. Get the complete data for the anchor sample
        anchor_data = self.anchor_dataset[idx]

        # 2. Get the corresponding positive images from each noisy dataset
        positive_images = []
        for noise in self.noise_types:
            positive_data = self.positive_datasets[noise][idx]
            positive_images.append(positive_data['image'])

        # 3. Add the list of positive images to the anchor's data dictionary
        anchor_data['positives'] = positive_images

        return anchor_data