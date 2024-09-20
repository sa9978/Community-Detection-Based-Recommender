from torch_geometric.datasets import Planetoid, Amazon, LastFMAsia, Coauthor

def load_dataset(dataset_name):
    if dataset_name == 'cora':
        # Cora: 2,708 nodes, 5,429 edges
        return Planetoid(root='./Data/Cora', name='Cora')
    elif dataset_name == 'citeseer':
        # Citeseer: 3,327 nodes, 4,732 edges
        return Planetoid(root='./Data/Citeseer', name='Citeseer')
    elif dataset_name == 'pubmed':
        # PubMed: 19,717 nodes, 44,338 edges
        return Planetoid(root='./Data/PubMed', name='PubMed')
    elif dataset_name == 'lastfmasia':
        # LastFMAsia: 7,624 nodes, 55,612 edges
        return LastFMAsia(root='./Data/LastFMAsia')
    elif dataset_name == 'amazon_photos':
        # Amazon Photos: 7,487 nodes, 119,043 edges
        return Amazon(root='./Data/Amazon', name='Photo')
    elif dataset_name == 'amazon_computers':
        # Amazon Computers: 13,752 nodes, 245,861 edges
        return Amazon(root='./Data/Amazon', name='Computers')
    elif dataset_name == 'coauthor_cs':
        # Coauthor CS: 18,333 nodes, 81,894 edges
        return Coauthor(root='./Data/CoauthorCS', name='CS')
    elif dataset_name == 'coauthor_physics':
        # Coauthor Physics: 34,493 nodes, 247,962 edges
        return Coauthor(root='./Data/CoauthorPhysics', name='Physics')
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized. Available options are: cora, citeseer, pubmed, lastfmasia, amazon_photos, amazon_computers, coauthor_cs, coauthor_physics.")

if __name__ == "__main__":
    for ds in ["lastfmasia", "coauthor_cs", "coauthor_physics", "amazon_computers", "amazon_photos"]:
        load_dataset(ds)
    # load_dataset("amazon_photos")