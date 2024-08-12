import tqdm
import time

for x in tqdm.tqdm(range(10)):
    time.sleep(0.5)
    1/0
    print("HEEEELLLLOOOO")