import gmm_data as gd

all_data = gd.read_data(gmm_data.root2 + 'CelebData/RSTrain/Points/', 45)

representation = gd.prepare_bird_representation(all_data)

print gd.fit_bird_GMM(representation)
