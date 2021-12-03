from typing import List, Tuple, Dict

# The alphabet for our character-based model
LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', \
                'l', 'm', 'n', 'o', 'p', 'q','r', 's', 't', 'u', 'v', 'w', 'x', \
                'y', 'z', '-', "'", '.', '_', '+', ' ', '<sos>', '<eos>']
                
def create_dictionaries(letter_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    '''
    Create dictionaries for letter2index and index2letter transformations
    '''
    letter2index, index2letter = {}, {}
    for index, char in enumerate(letter_list):
        index2letter[index] = char
        letter2index[char] = index
    return letter2index, index2letter
    
def transform_letter_to_index(raw_transcripts: List[str]) -> List[int]:
    '''
    Transforms text input to numerical input by converting each letter
    to its corresponding index from letter_list
    '''
    return [letter2index[char] for char in raw_transcripts]
    
letter2index, index2letter = create_dictionaries(LETTER_LIST)
