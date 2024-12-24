import sys
import inference as inf # type:ignore
from pathlib import Path
# sys.path.append('/Users/ajitj/Google Drive/ImageAnalysis/cell_analysis')
sys.path.append('')
model_name = 'HeLa' # can be on of ['Hela', 'U2OS']
confluency_est = 1800 # can be in the interval (0, 2000]
conf_threshold = .275 # can be in the interval (0, 1)
movie_file = Path('20241205_Bub1 ppg1 ppg2 pps121_F12_s3_phs.tif')
interval = [0, 1]

def main():
    container = inf.configure(model_name, confluency_est, conf_threshold)
    result = inf.run_inference(container, movie_file, interval)

    print(result['name'])
    print(result['semantic_movie'].shape)
    print(result['instance_movie'].shape)
    print(result['centroids'].shape)
    print(len(result['confidence']))

    return result

if __name__ == "__main__":
    result = main()
