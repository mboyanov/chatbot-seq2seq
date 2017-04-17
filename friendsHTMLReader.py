from datasetReader import DatasetReader
from bs4 import BeautifulSoup
import os


class FriendsHTMLReader(DatasetReader):

    def isLine(self, text):
        return len(text) > 0 and text[0] not in ['[', '('] and ":" in text[:min(20, len(text))]

    def words(self, data_path, tokenizer, exclude_fn=lambda x: False):
        raise NotImplemented

    def conversations(self, data_path, exclude_fn=lambda x: False, yield_fn=lambda x: (x[6], x[7])):
        data_dir = os.path.dirname(data_path)
        for file in os.listdir(data_dir):
            if file.endswith(".txt"):
                with open(os.path.join(data_dir,file)) as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')

                paras = soup.find_all('p')
                current_conversation = []
                for para in paras:
                    if self.isLine(para.text):
                        current_conversation.append(para.text)
                    else:
                        for i in range(len(current_conversation)-1):
                            yield((current_conversation[i], current_conversation[i+1]))
                        current_conversation = []
