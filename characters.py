import json
from settings import CHAR_TO_NUM_JSON

PAD_TOKEN_IDX = 59
START_TOKEN_IDX = 60
END_TOKEN_IDX = 61

PAD_TOKEN = 'P'
START_TOKEN = '<'
END_TOKEN = '>'

# Build char_to_num
with open(CHAR_TO_NUM_JSON, "r") as f:
    char_to_num = json.load(f)

char_to_num[PAD_TOKEN] = PAD_TOKEN_IDX  # pad token
char_to_num[START_TOKEN] = START_TOKEN_IDX  # start token
char_to_num[END_TOKEN] = END_TOKEN_IDX  # end token

# Build num_to_char
num_to_char = {j: i for i, j in char_to_num.items()}

if __name__ == "__main__":
    print(f'{char_to_num = }')
    print()
    print(f'{num_to_char = }')
