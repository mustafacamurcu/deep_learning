import gmm_data as gd

all_data = gd.read_data(gd.root2 + 'BirdData/RSTrain/Points/', 45)

representation = gd.prepare_bird_representation(all_data)

print gd.fit_bird_GMM(representation)
