import json
from collections import Counter
from pathlib import Path
from typing import Union

import arabic_reshaper
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from hazm import Normalizer, word_tokenize
from loguru import logger
from src.data import DATA_DIR
from wordcloud import WordCloud


class ChatStatistics:
    """Generates chat statistics from a telegram chat json file
    """
    def __init__(self, chat_json: Union[str, Path]):
        """
        :param chat_json: path to telegram json file
        """

        # load chat_data
        logger.info(f"loadind chat data from {chat_json}")
        with open(chat_json) as f:
            self.chat_data = json.load(f)

        self.normalizer = Normalizer()
        
        # load stop_words
        logger.info(f"loading stop words from {DATA_DIR / 'stopwords.txt'}")
        stop_words = open(DATA_DIR / 'stopwords.txt').readlines()
        stop_words = list(map(str.strip, stop_words))
        self.stop_words = list(map(self.normalizer.normalize, stop_words))

    def generate_wordcloud(self, output_dir):
        """
        :param output_dir: path to output directory for word cloud image
        """

        logger.info(f"loading text content...")
        text_content = ''
        for msg in self.chat_data['messages']:
            if type(msg['text']) is str:
                tokens = word_tokenize(msg['text'])
                tokens = list(filter(lambda item: item not in self.stop_words, tokens))
                text_content += f"{' '.join(tokens)}"

        # Normalize, reshape for final word cloud
        text_content = self.normalizer.normalize(text_content)  
        text_content = arabic_reshaper.reshape(text_content)
        text_content = get_display(text_content)  

        # generate word cloud
        logger.info(f"generating word cloud...")
        wordcloud = WordCloud(
            font_path=str(DATA_DIR / 'BHoma.ttf'), 
            background_color='white'
            ).generate(text_content)

        logger.info(f"saving word cloud to {output_dir}")
        wordcloud.to_file(str(Path(output_dir) / 'wordcloud.png'))  

if __name__ == "__main__":
    chat_stats = ChatStatistics(chat_json=(DATA_DIR / 'online.json'))
    chat_stats.generate_wordcloud(output_dir=DATA_DIR)
    print('Done!')
