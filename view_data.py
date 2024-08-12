import argparse
import pandas as pd

from settings import TRAIN_META, TRAIN_DIR

metadata_df = pd.read_csv(TRAIN_META)

class SequenceSample():
    def __init__(self, sample_index):
        sequence_df = metadata_df.iloc[sample_index]
        self.name = f'Sequence {sample_index:04d}'
        self.sequence_id = sequence_df['sequence_id']
        self.file_id = sequence_df['file_id']
        self.phrase = sequence_df['phrase']

    def get_path(self):
        return os.path.join(TRAIN_DIR, f'{self.file_id}.parquet')

    def get_df(self):
        # 獲取 Parquet 檔案的路徑
        pq_source = self.get_path()
        
        # 設置過濾條件，以過濾出對應 sequence_id 的數據
        pq_filter = [[('sequence_id', '=', self.sequence_id)],]
        
        # 讀取並過濾 Parquet 檔案，返回 Pandas DataFrame
        pq_table = pq.read_table(pq_source, filters = pq_filter)
        return pq_table.to_pandas()

if __name__ = '__name__':
    sample_sequence = SequenceSample(0)
    
    
    sample_sequence.__dict__