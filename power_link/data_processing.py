import torch
import sklearn.utils


def get_test_triplets(kg_data, device, shuffle=True, seed=None):
    """Extract test triplets to explain from a pre-processed KG dataset.

    Reads ``kg_data['test_tail']`` (the per-head ground-truth tail-prediction
    split produced by ``power_link.kge.load_kg.process``) and returns three
    tensors of shape ``[N]``: head, relation, tail.

    Parameters
    ----------
    kg_data : dict
        The output of ``power_link.kge.load_kg.process``.
    device : torch.device
    shuffle : bool, default ``True``
        Shuffle the triplets so iteration order is independent of disk order.
    seed : int, optional
        If given, the shuffle is deterministic. Required for reproducible runs.
    """
    kg_data_test = kg_data['test_tail']
    triplets_test = []
    true_tails_test = []
    for ele in kg_data_test:
        triplet = list(ele['triple'])
        triplets_test.append(triplet)
        true_tails_test.append(triplet[-1])
    triplets_test = torch.tensor(triplets_test).to(device)
    true_tails_test = torch.tensor(true_tails_test).to(device)
    subj, rel, obj = triplets_test[:, 0], triplets_test[:, 1], true_tails_test
    if shuffle:
        subj, rel, obj = sklearn.utils.shuffle(subj, rel, obj, random_state=seed)
    return subj, rel, obj
